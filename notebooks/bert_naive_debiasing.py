#!/usr/bin/env python
# coding: utf-8

# Simply debias BERT by optimizing the log odds ratio

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
import matplotlib.pyplot as plt
from overrides import overrides
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import sys
sys.path.append("../lib")


# In[ ]:


lr = 1e-5
weight_reg = 1e-4
train_file = "gender_occ_pos_w_probs_train.txt"
val_file = "gender_occ_pos_w_probs_val.txt"


# In[ ]:


from bert_utils import Config, BertPreprocessor
config = Config(
    model_type="bert-base-uncased",
    max_seq_len=24,
    batch_size=32,
    bias_weight=1., # technically unnecessary, but for easier debugging
    consistency_weight=1.,
    lr=lr,
    weight_reg=weight_reg,
    disable_dropout=True,
    init_probs_precomputed=True,
    testing=True,
    remove_prior_bias=True,
    epochs=3,
    train_file=train_file,
    val_file=val_file,
)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


T = TypeVar("T")
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


# In[ ]:


processor = BertPreprocessor(config.model_type, config.max_seq_len)


# In[ ]:


DATA_ROOT = Path("../data")
MODEL_SAVE_DIR = Path("../weights")


# Read the model in here

# In[ ]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
masked_lm = BertForMaskedLM.from_pretrained(config.model_type)
masked_lm.eval()


# In[ ]:





# Dropout might be causing the model to be more uncertain, attributing lower probs to the correct sentence: disabling might help with logit explosion

# In[ ]:


if config.disable_dropout:
    def disable_dropout(mod):
        if hasattr(mod, "named_children"):
            for nm, child in mod.named_children():
                if "dropout" in nm: child.p = 0. # forcibly set to 0
                disable_dropout(child)
    disable_dropout(masked_lm)


# In[ ]:


masked_lm


# Freeze positional embeddings

# In[ ]:


masked_lm.bert.embeddings.position_embeddings.requires_grad = False
masked_lm.bert.embeddings.token_type_embeddings.requires_grad = False


# Freeze layer norm

# In[ ]:


for k, v in masked_lm.named_parameters():
    if "LayerNorm" in k: v.requires_grad = False


# In[ ]:





# In[ ]:





# # The Dataset

# In[ ]:


from allennlp.data.token_indexers import PretrainedBertIndexer

def flatten(x: List[List[T]]) -> List[T]:
        return [item for sublist in x for item in sublist]

token_indexer = PretrainedBertIndexer(
    pretrained_model=config.model_type,
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )

def tokenizer(s: str):
    maxlen = config.max_seq_len - 2
    toks = token_indexer.wordpiece_tokenizer(s)[:maxlen]
    return toks


# In[ ]:


def to_np(t): return t.detach().cpu().numpy()

def to_words(arr):
    if len(arr.shape) > 1:
        return [to_words(a) for a in arr]
    else:
        arr = to_np(arr)
        return " ".join([itot(i) for i in arr])


# In[ ]:


rev_vocab = {v: k for k, v in token_indexer.vocab.items()}

def ttoi(t: str): return token_indexer.vocab[t]
def itot(i: int): return rev_vocab[i]


# In[ ]:


from allennlp.data.vocabulary import Vocabulary
global_vocab = Vocabulary()


# In[ ]:





# ### Dataset

# In[ ]:


import csv
from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import (TextField, SequenceLabelField, LabelField, 
                                  MetadataField, ArrayField)

class BertTextField(TextField):
    @overrides
    def get_padding_lengths(self): # consistent padding lengths
        pad_lengths = super().get_padding_lengths()
        for k in pad_lengths.keys():
            pad_lengths[k] = config.max_seq_len
        return pad_lengths

class LongArrayField(ArrayField):
    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        tensor = torch.from_numpy(self.array)
        return tensor
    
class FloatArrayField(ArrayField):
    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.FloatTensor:
        tensor = torch.FloatTensor(self.array)
        return tensor

class DebiasingDatasetReader(DatasetReader):
    def __init__(self, tokenizer, token_indexers, 
                 init_probs_precomputed: bool=False,
                 remove_prior_bias: bool=False,
                 ) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.vocab = token_indexers["tokens"].vocab
        self._init_probs_precomputed = init_probs_precomputed
        self._remove_prior_bias = remove_prior_bias

    def _proc(self, x):
        if x == "[MASK]" or x == "[PAD]": return x
        else: return x.lower()
        
    @overrides
    def text_to_instance(self, tokens: List[str], w1: str, w2: str, 
                         p1: Optional[float], p2: Optional[float],
                         desired_bias: Optional[float],
                        ) -> Instance:
        fields = {}
        input_toks = [Token(self._proc(x)) for x in tokens]
        fields["input"] = BertTextField(input_toks, self.token_indexers)        
        # take [CLS] token into account
        mask_position = tokens.index("[MASK]") + 1
        fields["mask_positions"] = LongArrayField(
            np.array(mask_position, dtype=np.int64),
         )
        fields["target_ids"] = LongArrayField(np.array([
            self.vocab[w1], self.vocab[w2],
        ], dtype=np.int64))
                
        if self._init_probs_precomputed:
            fields["initial_prob_sum"] = FloatArrayField(np.array(p1 + p2, dtype=np.float32))
        else:
            with torch.no_grad():
                bert_input = (self.token_indexers["tokens"]
                              .tokens_to_indices(input_toks, global_vocab, "tokens"))
                token_ids = torch.LongTensor(bert_input["tokens"]).unsqueeze(0)
                probs = masked_lm(token_ids)[0, mask_position, :].detach().numpy()
                probs = (probs - probs.max())
                probs = probs.exp() / probs.exp().sum()
                fields["initial_prob_sum"] =                     FloatArrayField(np.array(probs[self.vocab[w1]] + probs[self.vocab[w2]],
                               dtype=np.float32))
            
        if self._remove_prior_bias:
            fields["desired_bias"] =                 FloatArrayField(np.array(desired_bias, dtype=np.float32))
        
        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        p1, p2 = 0., 0.
        with open(file_path, "rt") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if self._init_probs_precomputed: 
                    sentence, w1, w2, tgt, p1, p2, prior_bias, bias_score = row
                else: sentence, w1, w2, tgt = row
                yield self.text_to_instance(
                    self.tokenizer(sentence), 
                    w1, w2, # words
                    float(p1), float(p2), # initial probs
                    float(prior_bias), # prior bias
                )


# In[ ]:


reader = DebiasingDatasetReader(tokenizer=tokenizer, 
                                token_indexers={"tokens": token_indexer},
                                init_probs_precomputed=config.init_probs_precomputed,
                                remove_prior_bias=config.remove_prior_bias)
train_ds, val_ds = (reader.read(DATA_ROOT / fname) for fname in [config.train_file, config.val_file])


# In[ ]:


vars(train_ds[0].fields["input"])


# In[ ]:





# ### Data Iterator

# In[ ]:


from allennlp.data.iterators import BasicIterator

iterator = BasicIterator(
        batch_size=config.batch_size, 
    )
iterator.index_with(global_vocab)


# Sanity check

# In[ ]:


batch = next(iter(iterator(train_ds)))


# In[ ]:


batch


# In[ ]:





# # Model and Loss

# ### The loss function

# In[ ]:


def mse_loss(x, y, desired=0): return ((x - y - desired) ** 2).mean()
def mae_loss(x, y, desired=0): return (x - y - desired).abs().mean()
class HingeLoss(nn.Module):
    def __init__(self, margin: float=0.1):
        super().__init__()
        self.margin = margin
    def forward(self, x, y, desired=0.):
        return torch.relu((x - y - desired).abs().mean() - self.margin)


# In[ ]:


def neg_likelihood(ll, # (batch, )
               initial_prob_sum, # (batch, )
     ):
    """log likelihood of either of the target ids being chosen"""
    return -ll.mean()

class LogitConsistency(nn.Module):
    def __init__(self, distance: Callable):
        super().__init__()
        self._distance = distance
    
    def forward(self, ll, # (batch, )
                initial_prob_sum, # (batch, )
               ):
        """
        Constrains prob sum put on two words to be roughly equal
        TODO: Provide some probabilistic/statistical interpretation
        """
        d = self._distance(ll, initial_prob_sum.log())
        return d


# In[ ]:


from allennlp.training.metrics import Metric
class TotalProbDiff(Metric):
    def __init__(self):
        super().__init__()
        self._total = 0
        self._n_obs = 0
        
    def __call__(self, ll, initial_prob_sum):
        self._total += (ll.exp() - initial_prob_sum).mean().item()
        self._n_obs += 1
        
    def get_metric(self, reset: bool=False):
        mtrc = self._total / self._n_obs
        if reset: self.reset()
        return mtrc
    
    def reset(self):
        self._total = 0
        self._n_obs = 0


# In[ ]:


class BiasLoss(nn.Module):
    """
    Returns the deviation of the log odds ratio from its desired value.
    Denoting the probs as p and q there are several options available:
        - MSE(log p, log q)
        - Max-margin loss
    Most processing takes place here because there is a lot of shared heavy processing required
    (e.g. computing partition function)
    TODO: Add option to set the optimal log odds ratio
    TODO: Ensure the logits do not change significantly
    """
    def __init__(self, loss_func: Callable=mae_loss,
                 consistency_loss_func: Callable=LogitConsistency(mae_loss),
                 bias_weight: float=1.,
                 consistency_weight: float=1.):
        super().__init__()
        self.loss_func = loss_func
        self._consistency_loss = consistency_loss_func
        self.consistency_weight = consistency_weight
        self.bias_weight = bias_weight
        self._total_prob_diff = TotalProbDiff()
    
    @staticmethod
    def _log_likelihood(logits, # (batch, V)
                        target_logits, # (batch, )
                       ) -> torch.FloatTensor: # (batch, )
        max_logits = logits.max(1, keepdim=True)[0] # (batch, )
        log_exp_sum_logits = ((logits - max_logits).exp()
                              .sum(1).log()) # (batch, )
        # these logits should never be masked
        log_exp_sum_correct_logits = ((target_logits - max_logits).exp()
                                      .sum(1).log()) # (batch, )
        return log_exp_sum_correct_logits - log_exp_sum_logits
        
    def forward(self, logits: torch.FloatTensor, # (batch, seq, V)
                mask_positions: torch.LongTensor, # (batch, )
                target_ids: torch.LongTensor, # (batch, 2)
                initial_prob_sum: torch.FloatTensor, # (batch, )
                desired_bias: torch.FloatTensor=None,
               ) -> torch.FloatTensor:
        """
        input_ids: Numericalized tokens
        mask_position: Positions of mask tokens
        target_ids: Ids of target tokens to compute log odds on
        padding_mask: padding positions
        """
        bs, seq = logits.size(0), logits.size(1)

        # Gather the logits for at the masked positions
        # TODO: More efficient implementation?
        # Gather copies the data to create a new tensor which we would rather avoid
        sel = (mask_positions.unsqueeze(1)
                .unsqueeze(2).expand(bs, 1, logits.size(2))) # (batch, 1, V)
        logits_at_masked_positions = logits.gather(1, sel).squeeze(1) # (batch, V)
        
        # Gather the logits for the target ids
        sel = target_ids
        target_logits_at_masked_positions = logits_at_masked_positions.gather(1, sel).squeeze(1) # (batch, 2)
        
        bias_loss = self.loss_func(
            target_logits_at_masked_positions[:, 0], # male logits
            target_logits_at_masked_positions[:, 1], # female logits
            desired=desired_bias if desired_bias is not None else 0.,
         )
        
        # compute log likelihood of either of the target ids being observed
        ll = self._log_likelihood(logits_at_masked_positions,
                                  target_logits_at_masked_positions)
        
        # enforce consistency between prior probabilities and current probabilities
        consistency_loss = self._consistency_loss(
            ll, initial_prob_sum,
         )
        out_dict = {}
        out_dict["bias_loss"] = bias_loss * self.bias_weight
        out_dict["consistency_loss"] = consistency_loss * self.consistency_weight
        out_dict["loss"] = out_dict["bias_loss"] + out_dict["consistency_loss"]
        out_dict["total_prob_diff"] = self._total_prob_diff(ll, initial_prob_sum)
        return out_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"total_prob_diff": self._total_prob_diff.get_metric(reset)}


# Sanity checks

# In[ ]:


to_words(batch["input"]["tokens"])


# In[ ]:


batch


# In[ ]:


bias_loss = BiasLoss()
logits = masked_lm(batch["input"]["tokens"])
bias_loss(logits, batch["mask_positions"], batch["target_ids"],
          batch["initial_prob_sum"])


# In[ ]:


probs = torch.softmax(logits[:, 1, :], 1)


# In[ ]:


sentence, w1, w2 = "[MASK] is a nurse", "he", "she"

tokens = tokenizer(sentence)
mask_position = tokens.index("[MASK]") + 1
input_toks = [Token(w) for w in tokens]
bert_input = (token_indexer.tokens_to_indices(input_toks, global_vocab, "tokens"))
token_ids = torch.LongTensor(bert_input["tokens"]).unsqueeze(0)


# In[ ]:


probs[:, token_indexer.vocab[w1]] + probs[:, token_indexer.vocab[w2]]


# In[ ]:


batch["initial_prob_sum"]


# In[ ]:





# ### The allennlp model (for training)

# In[ ]:


from copy import deepcopy


# In[ ]:


from allennlp.models import Model

class BERT(Model):
    def __init__(self, vocab, bert_for_masked_lm, 
                 loss: nn.Module=BiasLoss()):
        super().__init__(vocab)
        self.bert_for_masked_lm = bert_for_masked_lm
        self.loss = loss
    
    def forward(self, 
                input: TensorDict,
                mask_positions: torch.LongTensor,
                target_ids: torch.LongTensor,
                initial_prob_sum: torch.FloatTensor,
                desired_bias: torch.FloatTensor=None,
            ) -> TensorDict:
        logits = self.bert_for_masked_lm(input["tokens"])
        # most of processing takes place in loss func
        out_dict = self.loss(logits, mask_positions, 
                             target_ids, initial_prob_sum,
                             desired_bias=desired_bias,
                            )
        out_dict["logits"] = logits
        return out_dict
    
    def get_metrics(self, reset: bool=False):
        return self.loss.get_metrics()


# In[ ]:


logit_distance = mae_loss

loss = BiasLoss(
    loss_func=logit_distance,
    consistency_loss_func=LogitConsistency(logit_distance),
    bias_weight=config.bias_weight,
    consistency_weight=config.consistency_weight,
)
model = BERT(global_vocab, masked_lm, loss=loss)


# In[ ]:


init_dict = dict(model.state_dict())


# In[ ]:


model.load_state_dict(init_dict)


# In[ ]:


orig_weights = {k: deepcopy(v) for k, v in model.named_parameters()}


# In[ ]:





# ### Bias scores before

# In[ ]:


masked_lm.eval()
logits = masked_lm(processor.to_bert_model_input("[MASK] is a housemaid"))[0, 1]


# In[ ]:


logits[ttoi("he")]


# In[ ]:


logits[ttoi("she")]


# In[ ]:


probs = torch.softmax(logits.unsqueeze(0), 1).squeeze(0)


# In[ ]:


probs[ttoi("he")]


# In[ ]:


probs[ttoi("she")]


# For word not in vocab

# In[ ]:


logits = masked_lm(processor.to_bert_model_input("[MASK] is a slut"))[0, 1]


# In[ ]:


logits[ttoi("he")]


# In[ ]:


logits[ttoi("she")]


# In[ ]:


probs = torch.softmax(logits.unsqueeze(0), 1).squeeze(0)


# In[ ]:


probs[ttoi("he")]


# In[ ]:


probs[ttoi("she")]


# In[ ]:





# In[ ]:





# ### Probability distribution for unrelated sentence

# In[ ]:


def print_topk_preds(masked_sentence, k=5, strlen=30):
    mask_idx = [x.text for x in processor.tokenize(masked_sentence)].index("[MASK]") + 1
    logits = masked_lm(processor.to_bert_model_input(masked_sentence))
    probs = torch.softmax(logits.squeeze(0), 1)
    topk = []
    for p, id_ in zip(*probs[mask_idx, :].topk(k)):
        topk.append(("%.4f" % p.item(), itot(id_.item())))
    print("\n".join([f"{masked_sentence.replace('[MASK]', w)}:{' ' * (strlen - len(w) - len(p))}{p}" for p, w in topk]))


# In[ ]:


print_topk_preds("i ride my [MASK] to work")


# In[ ]:


print_topk_preds("the [MASK] wagged its tail")


# In[ ]:


print_topk_preds("the fish [MASK] through the water")


# In[ ]:





# # Confirming Bias Scores Before

# ### Train

# In[ ]:


def compute_bias_score(row):
    sentence, fword, mword, prior_bias = [row[k] for k in ["sentence", "fword", "mword", "prior_bias"]]
    mask_pos = tokenizer(sentence).index("[MASK]") + 1
    logits = masked_lm(processor.to_bert_model_input(sentence)).squeeze(0)
    i1,i2 = ttoi(fword),ttoi(mword)
    log_odds = logits[mask_pos, i1] - logits[mask_pos, i2]
    bias_correction = prior_bias
    return (log_odds - bias_correction).item()


# In[ ]:


df_train = pd.read_csv(DATA_ROOT / config.train_file)


# In[ ]:


plt.hist(df_train["original_bias_score"])


# In[ ]:





# ### Dev

# In[ ]:


df_val = pd.read_csv(DATA_ROOT / config.val_file)


# In[ ]:


plt.hist(df_val["original_bias_score"])


# In[ ]:





# # Training Loop

# In[ ]:


from allennlp.training import Callback


# In[ ]:


from copy import deepcopy

class StatisticRecorder(Callback):
    def __init__(self, orig_weights, rec_periods=1):
        self.rec_periods = rec_periods
        self.norms = {k: [] for k, v in model.named_parameters() if v.requires_grad}
        self.grad_magnitudes = {k: [] for k, v in model.named_parameters() if v.requires_grad}
        self._orig_weights = orig_weights
        self.change_magnitudes = {k: [] for k, v in model.named_parameters() if v.requires_grad}
        
    def on_batch_end(self, data):
        if (data['batches_this_epoch'] + 1) % self.rec_periods == 0:
            with torch.no_grad():
                for k, p in self.trainer.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        Z = torch.norm(p).item()
                        self.norms[k].append(Z)
                        self.grad_magnitudes[k].append((torch.norm(p.grad) / Z).item())
                        self.change_magnitudes[k].append((torch.norm(p - self._orig_weights[k]) / Z).item())


# In[ ]:


class WeightDeviationRegularizor(Callback):
    def __init__(self, orig_weights, weight=1e-4, l1=True):
        self.orig_weights = orig_weights
        self.weight = weight
        self.l1 = l1
        
    def get_reg_term(self, now, orig):
        if self.l1:
            return torch.where(now < orig, torch.ones_like(now), -torch.ones_like(now))
        else:
            return (orig - now)
        
    def on_backward_end(self, data):
        lr = config.lr
        with torch.no_grad():
            for name, param in self.trainer.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    reg_term = self.weight * lr * self.get_reg_term(param.data, orig_weights[name])
                    param.data.add_(reg_term)


# In[ ]:


from collections import defaultdict
class LossMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.logs = defaultdict(list)
    def on_forward_end(self, payload):
        for k, v in payload.items():
            if "loss" in k: self.logs[k].append(v.item())


# In[ ]:


stat_rec = StatisticRecorder(orig_weights, rec_periods=1)
wdd = WeightDeviationRegularizor(orig_weights, weight=config.weight_reg)
monitor = LossMonitor()


# In[ ]:


def use(name: str):
    if "LayerNorm" in name: return False
    if "position_embeddings" in name: return False
    if "token_type" in name: return False
    return True


# In[ ]:


filtered_params = [p for name, p in model.named_parameters() if use(name)]


# In[ ]:


optimizer = torch.optim.Adam(filtered_params, lr=config.lr, weight_decay=0.)


# In[ ]:


from allennlp.training.learning_rate_schedulers import SlantedTriangular, CosineWithRestarts
# use slanted triangular lr scheduler to prevent initial spike in consistency loss
lr_sched = SlantedTriangular(optimizer, 
                             num_epochs=config.epochs, 
                             num_steps_per_epoch=iterator.get_num_batches(train_ds))


# In[ ]:


from allennlp.training import TrainerWithCallbacks

trainer = TrainerWithCallbacks(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    validation_dataset=val_ds,
    callbacks=[stat_rec, wdd, monitor],
    learning_rate_scheduler=lr_sched,
    #     serialization_dir=DATA_ROOT / "debias_ckpts",
    cuda_device=0 if torch.cuda.is_available() else -1,
    num_epochs=config.epochs,
)


# In[ ]:


trainer.train()


# In[ ]:





# In[ ]:





# Analyzing changes

# In[ ]:


change_sorted_weights = sorted([(-v[-1], k) for k, v in stat_rec.change_magnitudes.items() if len(v) > 0])
{k.replace("bert_for_masked_lm.bert.encoder.", ""): -x for x, k in change_sorted_weights}


# In[ ]:


n = 10
fig = plt.figure(figsize=(10, n * 4))
for i, (_, k) in enumerate(change_sorted_weights[:n]):
    ax = fig.add_subplot(n, 1, i+1)
    ax.plot(stat_rec.grad_magnitudes[k])


# In[ ]:





# ### Change in loss breakdown

# In[ ]:


fig = plt.figure(figsize=(10, 30))
ax = fig.add_subplot(n, 1, 1)
ax.plot(monitor.logs["bias_loss"], label="bias loss")
ax.plot(monitor.logs["consistency_loss"], label="consistency loss")
ax.legend()


# In[ ]:





# In[ ]:





# # Evaluate

# Simple prediction

# In[ ]:


def get_preds(model, batch: TensorDict):
    return model(**batch)["logits"].argmax(2)


# In[ ]:


to_words(batch["input"]["tokens"])


# In[ ]:


to_words(get_preds(model, batch))


# In[ ]:





# ### Logits and bias

# In[ ]:


masked_lm.eval()
logits = masked_lm(processor.to_bert_model_input("[MASK] is a housemaid"))[0, 1]


# In[ ]:


logits[ttoi("he")]


# In[ ]:


logits[ttoi("she")]


# Probabilities

# In[ ]:


probs = torch.softmax(logits.unsqueeze(0), 1).squeeze(0)


# In[ ]:


probs[ttoi("he")]


# In[ ]:


probs[ttoi("she")]


# In[ ]:





# ##### For an example not in the vocabulary

# In[ ]:


logits = masked_lm(processor.to_bert_model_input("[MASK] is a slut"))[0, 1]


# In[ ]:


logits[ttoi("he")]


# In[ ]:


logits[ttoi("she")]


# In[ ]:


probs = torch.softmax(logits.unsqueeze(0), 1).squeeze(0)


# In[ ]:


probs[ttoi("he")]


# In[ ]:


probs[ttoi("she")]


# In[ ]:





# Changes to output distribution of unrelated sentences

# In[ ]:


print_topk_preds("i ride my [MASK] to work")


# In[ ]:


print_topk_preds("the [MASK] wagged its tail")


# In[ ]:


print_topk_preds("the fish [MASK] through the water")


# In[ ]:





# ### Evaluation on bias score across the train and val set

# In[ ]:


from tqdm import tqdm
tqdm.pandas()


# In[ ]:


df_train["bias_score_after"] = df_train.progress_apply(compute_bias_score, axis=1)


# In[ ]:


plt.hist(df_train["bias_score_after"])


# The decrease is smaller than expected: perhaps more training is necessary?

# In[ ]:


df_train["original_bias_score"].abs().mean()


# In[ ]:


df_train["bias_score_after"].abs().mean()


# In[ ]:





# In[ ]:


df_val["bias_score_after"] = df_val.progress_apply(compute_bias_score, axis=1)


# In[ ]:


df_val["original_bias_score"].abs().mean()


# In[ ]:


df_val["bias_score_after"].abs().mean()


# In[ ]:





# # Export Weights

# As PyTorch state dict

# In[ ]:


torch.save(masked_lm.state_dict(), MODEL_SAVE_DIR / "state_dict.pth")


# TODO: Export as tensorflow checkpoint?

# In[ ]:





# In[ ]:




