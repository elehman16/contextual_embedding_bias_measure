#!/usr/bin/env python
# coding: utf-8

# Precompute prior probabilities

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import csv
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


# In[3]:


import sys
sys.path.append("../lib")
DATA_ROOT = Path("../data")


# In[4]:


from bert_utils import Config, BertPreprocessor
config = Config(
    model_type="bert-base-uncased",
    max_seq_len=24,
    batch_size=64,
    consistency_weight=0.,
    prior_precomputed=True,
    testing=True,
)


# In[5]:


processor = BertPreprocessor(config.model_type, config.max_seq_len)


# In[6]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
masked_lm = BertForMaskedLM.from_pretrained(config.model_type)
masked_lm.eval()


# In[ ]:





# In[7]:


from allennlp.data import Token
from allennlp.data.token_indexers import PretrainedBertIndexer

token_indexer = PretrainedBertIndexer(
    pretrained_model=config.model_type,
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )
#     if len(toks) < config.max_seq_len:
#         return toks + (["[PAD]"] * (maxlen - len(toks)))
#     else:

def tokenizer(s: str):
    maxlen = config.max_seq_len - 2
    toks = token_indexer.wordpiece_tokenizer(s)[:maxlen]
    return toks


# In[8]:


from allennlp.data.vocabulary import Vocabulary
global_vocab = Vocabulary()


# In[ ]:





# In[9]:


in_file_path = DATA_ROOT / "sample.csv"
out_file_path = DATA_ROOT / "sample_w_probs.csv"


# ## Sample

# In[10]:


with open(in_file_path, "rt") as f:
    reader = csv.reader(f)
    sentence, w1, w2, tgt = next(iter(reader))


# In[11]:


sentence


# In[12]:


tokens = tokenizer(sentence); tokens


# In[13]:


input_toks = [Token(w) for w in tokens]; input_toks


# In[14]:


token_indexer.tokens_to_indices(input_toks, global_vocab, "tokens")


# In[15]:


bert_input = (token_indexer.tokens_to_indices(input_toks, global_vocab, "tokens"))


# In[16]:


token_ids = torch.zeros(1, config.max_seq_len, dtype=torch.long)
token_ids[0, :len(bert_input["tokens"])] = torch.LongTensor(bert_input["tokens"])


# In[17]:


token_ids


# In[18]:


tokens = tokenizer(sentence.replace(tgt, "[MASK]"))
mask_comes_before_tgt = (sentence.find("[MASK]") < sentence.find(tgt))
if mask_comes_before_tgt:
    mask_position = tokens.index("[MASK]") + 1
else:
    mask_position = len(tokens) - tokens[::-1].index("[MASK]")


# In[19]:


input_toks = [Token(w) for w in tokens]
bert_input = (token_indexer.tokens_to_indices(input_toks, global_vocab, "tokens"))
# pad to consistent length
token_ids = torch.zeros(1, config.max_seq_len, dtype=torch.long)
token_ids[0, :len(bert_input["tokens"])] = torch.LongTensor(bert_input["tokens"])
logits = masked_lm(token_ids)[0, mask_position, :]
l1, l2 = logits[token_indexer.vocab[w1]], logits[token_indexer.vocab[w2]]
desired_bias = (l1 - l2).item()


# In[20]:


desired_bias


# In[ ]:





# ## Actual Processing

# In[21]:


with open(in_file_path, "rt") as f:
    with open(out_file_path, "wt") as fout:
        reader = csv.reader(f)
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            sentence, w1, w2, tgt = row
                        
            # compute probabilities
            with torch.no_grad():
                tokens = tokenizer(sentence)
                mask_position = tokens.index("[MASK]") + 1
                input_toks = [Token(w) for w in tokens]
                bert_input = (token_indexer.tokens_to_indices(input_toks, global_vocab, "tokens"))
                # pad to consistent length
                token_ids = torch.zeros(1, config.max_seq_len, dtype=torch.long)
                token_ids[0, :len(bert_input["tokens"])] = torch.LongTensor(bert_input["tokens"])

                probs = torch.softmax(masked_lm(token_ids)[:, mask_position, :], 1).squeeze(0).detach().numpy()
                p1, p2 = probs[token_indexer.vocab[w1]], probs[token_indexer.vocab[w2]]
                
            # compute desired bias (=bias when word is masked)
            with torch.no_grad():
                tokens = tokenizer(sentence.replace(tgt, "[MASK]"))
                mask_comes_before_tgt = (sentence.find("[MASK]") < sentence.find(tgt))
                if mask_comes_before_tgt:
                    new_mask_position = tokens.index("[MASK]") + 1
                    assert new_mask_position == mask_position
                else:
                    mask_position = len(tokens) - tokens[::-1].index("[MASK]")
                input_toks = [Token(w) for w in tokens]
                
                bert_input = (token_indexer.tokens_to_indices(input_toks, global_vocab, "tokens"))
                # pad to consistent length
                token_ids = torch.zeros(1, config.max_seq_len, dtype=torch.long)
                token_ids[0, :len(bert_input["tokens"])] = torch.LongTensor(bert_input["tokens"])
                logits = masked_lm(token_ids)[0, mask_position, :]
                l1, l2 = logits[token_indexer.vocab[w1]], logits[token_indexer.vocab[w2]]
                desired_bias = (l1 - l2).item()
                
            writer.writerow([sentence, w1, w2, tgt, p1, p2, desired_bias]) 


# In[ ]:





# In[ ]:




