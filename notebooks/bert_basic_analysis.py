#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    model_type="bert-base-uncased",
    max_seq_len=128,
)


# In[3]:


T = TypeVar('T')
def flatten(x: List[List[T]]) -> List[T]:
    return [item for sublist in x for item in sublist]


# In[4]:


from allennlp.common.util import get_spacy_model
from spacy.attrs import ORTH
from spacy.tokenizer import Tokenizer

nlp = get_spacy_model("en_core_web_sm", pos_tags=False, parse=True, ner=False)
nlp.tokenizer.add_special_case("[MASK]", [{ORTH: "[MASK]"}])
def spacy_tok(s: str):
    return [w.text for w in nlp(s)]


# In[5]:


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers import Token

token_indexer = PretrainedBertIndexer(
    pretrained_model=config.model_type,
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )

# apparently we need to truncate the sequence here, which is a stupid design decision
def tokenize(x: str) -> List[Token]:
        return [Token(w) for w in flatten([
                token_indexer.wordpiece_tokenizer(w)
                for w in spacy_tok(x)]
        )[:config.max_seq_len]]


# In[6]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
model = BertForMaskedLM.from_pretrained(config.model_type)


# In[7]:


from allennlp.data import Vocabulary

vocab = Vocabulary()
token_indexer._add_encoding_to_vocabulary(vocab)


# In[8]:


def get_logits(input_sentence: str) -> torch.Tensor:
    input_toks = tokenize(input_sentence)
    batch = token_indexer.tokens_to_indices(input_toks, vocab, "tokens")
    token_ids = torch.LongTensor(batch["tokens"]).unsqueeze(0)
    with torch.no_grad():
        out_logits = model(token_ids).squeeze(0)
    return out_logits.detach().cpu().numpy()


# In[9]:


full_vocab = {v:k for k, v in token_indexer.vocab.items()}

def indices_to_words(indices: Iterable[int]) -> List[str]:
    return [full_vocab[x] for x in indices]


# In[10]:


indices_to_words(get_logits("he is very [MASK].").argmax(1))


# In[11]:


indices_to_words(get_logits("he is [MASK].").argmax(1))


# In[14]:


indices_to_words(get_logits("she is [MASK].").argmax(1))


# In[15]:


indices_to_words(get_logits("she is very [MASK].").argmax(1))


# In[ ]:





# The usual stuff

# In[16]:


indices_to_words(get_logits("[MASK] is a doctor.").argmax(1))


# In[17]:


indices_to_words(get_logits("[MASK] is a nurse.").argmax(1))


# In[18]:


indices_to_words(get_logits("[MASK] is a programmer.").argmax(1))


# In[ ]:





# Measuring difference

# In[19]:


male_logits = get_logits("he is very [MASK].")[4, :]
female_logits = get_logits("she is very [MASK].")[4, :]


# In[20]:


def softmax(x, axis=0, eps=1e-9):
    e = np.exp(x)
    return e / (e.sum(axis, keepdims=True) + eps)


# In[21]:


male_probs = softmax(male_logits)
female_probs = softmax(female_logits)


# In[22]:


msk = ((male_probs >= 1e-6) & (female_probs >= 1e-6))
male_probs = male_probs[msk]
female_probs = female_probs[msk]


# In[23]:


[(pos + 1, full_vocab[i]) for i, pos in enumerate((male_probs / female_probs).argsort()) if pos < 10]


# In[24]:


[(pos + 1, full_vocab[i]) for i, pos in enumerate((female_probs / male_probs).argsort()) if pos < 10]


# In[ ]:





# # Construct measure of bias

# In[25]:


input_sentence = "[MASK] is intelligent"


# In[48]:


def _get_mask_index(toks: Iterable[Token]) -> int:
    for i, t in enumerate(toks):
        if t.text == "[MASK]":
            return i + 1 # take the [CLS] token into account
    raise ValueError("No [MASK] token found")


# In[87]:


def get_logits(input_sentence: str, n_calc: int=10) -> np.ndarray:
    """
    n_calc: Since the logits are non-deterministic, 
    computing the logits multiple times might be better
    """
    input_toks = tokenize(input_sentence)
    batch = token_indexer.tokens_to_indices(input_toks, vocab, "tokens")
    token_ids = torch.LongTensor(batch["tokens"]).unsqueeze(0)
    
    logits = None
    for _ in range(n_calc):
        with torch.no_grad():
            out_logits = model(token_ids).squeeze(0)
        if logits is None: logits = np.zeros(out_logits.shape)
        logits += out_logits.detach().cpu().numpy()
    return logits / n_calc


# In[88]:


def get_logit_scores(input_sentence: str, words: int) -> Dict[str, float]:
    out_logits = get_logits(input_sentence)
    input_toks = tokenize(input_sentence)
    i = _get_mask_index(input_toks)
    return {w: out_logits[i, token_indexer.vocab[w]] for w in words}

def get_log_odds(input_sentence: str, word1: str, word2: str) -> float:
    scores = get_logit_scores(input_sentence, (word1, word2))
    return scores[word1] - scores[word2]


# In[89]:


get_logit_scores("[MASK] is intelligent.", ["she", "he"])


# In[90]:


get_log_odds("[MASK] is intelligent.", "she", "he")


# In[ ]:





# Surprisingly, marriage is more strongly associated with he than she

# In[91]:


get_log_odds("[MASK] is married.", "she", "he")


# In[92]:


get_log_odds("[MASK] is alive.", "she", "he")


# In[93]:


get_log_odds("[MASK] is a person.", "she", "he")


# In[94]:


get_log_odds("[MASK] is a doctor.", "she", "he")


# In[95]:


get_log_odds("[MASK] is my mother.", "she", "he")


# In[96]:


get_log_odds("[MASK] is my father.", "she", "he")


# This is strange...

# In[100]:


get_log_odds("[MASK] is female.", "she", "he")


# In[101]:


get_log_odds("[MASK] is ugly.", "she", "he")


# This is strange too...

# In[102]:


get_log_odds("[MASK] is male.", "she", "he")


# In[103]:


get_log_odds("[MASK] is a housewife", "she", "he")


# In[106]:


get_log_odds("[MASK] is a girl", "she", "he")


# In[ ]:




