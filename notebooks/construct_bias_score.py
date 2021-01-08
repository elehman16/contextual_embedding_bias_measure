#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import sys
sys.path.append("../lib")


# In[4]:


from bert_utils import Config, BertPreprocessor


# In[5]:


config = Config(
    model_type="bert-base-uncased",
    max_seq_len=128,
)


# In[6]:


processor = BertPreprocessor(config.model_type, config.max_seq_len)


# In[ ]:





# In[7]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
model = BertForMaskedLM.from_pretrained(config.model_type)
model.eval() # Important! Disable dropout


# In[8]:


def get_logits(sentence: str) -> np.ndarray:
    return model(processor.to_bert_model_input(sentence))[0, :, :].cpu().detach().numpy()


# In[9]:


def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)


# In[10]:


from collections import defaultdict

def get_mask_fill_logits(sentence: str, words: Iterable[str],
                         use_last_mask=False, apply_softmax=False) -> Dict[str, float]:
    mask_i = processor.get_index(sentence, "[MASK]", last=use_last_mask)
    logits = defaultdict(list)
    out_logits = get_logits(sentence)
    if apply_softmax: 
        out_logits = softmax(out_logits)
    return {w: out_logits[mask_i, processor.token_to_index(w)] for w in words}


# Here, we will consider the "bias" of word $ w $ to be the difference in the strength of association of $ w $ with certain groups. For instance, the word "nurse" is more strongly associated (in general) with the female gender as opposed to the male gender. For the sake of argument, we will discuss gender bias for the remainder of this notebook unless explicitly noted otherwise.
# 
# There are two ways of measuring bias via the language model probabilities. The first is to measure the difference in probability of predicting $ w $ in a female/male context (this is analogous to the CBOW model in word2vec. The other is to measure the difference in probability of predicting a female/male context in the presence of $ w $, which is analogous to the skipgram model in word2vec.
# 
# We will denote the first difference as the *target fill bias* and the latter as the *context fill bias* (temporary terms). We measure the difference in probability by the log odds ratio. 
# 
# We want to measure the conditional probabilities in both cases (with the condition being either the word $ w $ or the context), so we need to correct for differences in prior probabilities. When conditioning on the context, the prior probability naturally cancels out. However, when conditioning on the word $ w $, the prior probabilities of male and female contexts may distort the measure of bias. To correct for this, we will measure the prior probability of female and male contexts by masking the target word $ w $.

# In[11]:


def bias_score(sentence: str, gender_words: Iterable[str], 
               word: str, gender_comes_first=True) -> Dict[str, float]:
    """
    Input a sentence of the form "GGG is XXX"
    XXX is a placeholder for the target word
    GGG is a placeholder for the gendered words (the subject)
    We will predict the bias when filling in the gendered words and 
    filling in the target word.
    
    gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
    """
    # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
    mw, fw = gender_words
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]
    # male words are simply more likely than female words
    # correct for this by masking the target word and measuring the prior probabilities
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction = subject_fill_prior_logits[mw] -                                             subject_fill_prior_logits[fw]
    
    # probability of filling "programmer" into [MASK] when subject is male/female
    try:
        mw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", mw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        fw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", fw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        # We don't need to correct for the prior probability here since the probability
        # should already be conditioned on the presence of the word in question
        tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
    except:
        tgt_fill_bias = np.nan # TODO: handle multi word case
    return {"gender_fill_bias": subject_fill_bias,
            "gender_fill_prior_correction": subject_fill_bias_prior_correction,
            "gender_fill_bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
            "target_fill_bias": tgt_fill_bias, 
           }


# In[12]:


get_mask_fill_logits("[MASK] is a nurse", ["she", "he"])


# ### Professions and nouns

# In[13]:


bias_score("GGG is a XXX.", ["he", "she"], "nurse")


# In[14]:


bias_score("GGG is a XXX.", ["he", "she"], "programmer")


# Looks like men are more likely to be dogs???

# In[15]:


bias_score("GGG is a XXX.", ["he", "she"], "dog")


# In[16]:


bias_score("GGG is a XXX.", ["he", "she"], "pig")


# In[17]:


bias_score("GGG is your XXX.", ["he", "she"], "father")


# In[18]:


bias_score("GGG is your XXX.", ["he", "she"], "mother")


# In[19]:


bias_score("GGG is wearing a XXX.", ["he", "she"], "dress")


# In[20]:


bias_score("GGG is wearing a XXX.", ["he", "she"], "shirt")


# In[21]:


bias_score("GGG is wearing a XXX.", ["he", "she"], "suit")


# In[ ]:





# One limitation of the target fill bias is that it cannot handle multiple words/wordpieces in its naive formulation...

# In[22]:


bias_score("GGG is a XXX.", ["he", "she"], "housemaid")


# Chair seems relatively neutral

# In[23]:


bias_score("GGG is a XXX.", ["he", "she"], "chair")


# Interestingly, "married" occurs with a much higher likelihood in male contexts...

# In[24]:


bias_score("GGG is XXX.", ["he", "she"], "married")


# In[ ]:





# ### Adjectives

# In[25]:


bias_score("GGG is very XXX.", ["he", "she"], "beautiful")


# In[26]:


bias_score("GGG is very XXX.", ["he", "she"], "violent")


# In[27]:


bias_score("GGG is very XXX.", ["he", "she"], "intelligent")


# In[28]:


bias_score("GGG is very XXX.", ["he", "she"], "normal")


# In[29]:


bias_score("GGG is very XXX.", ["he", "she"], "abnormal")


# In[30]:


bias_score("GGG is very XXX.", ["he", "she"], "sexy")


# In[31]:


bias_score("GGG is very XXX.", ["he", "she"], "cute")


# This is pretty weird...

# In[75]:


bias_score("GGG is XXX.", ["he", "she"], "male")


# In[76]:


bias_score("GGG is XXX.", ["he", "she"], "female")


# In[ ]:





# ### Other stuff

# Surprisingly, "she likes math" is considered more likely that "he likes math"

# In[32]:


bias_score("GGG likes XXX.", ["he", "she"], "math")


# Here, the subject and target biases disagree...

# In[33]:


bias_score("GGG likes XXX.", ["he", "she"], "science")


# Here, the subject and target biases disagree even more...

# In[34]:


bias_score("GGG enjoys XXX.", ["he", "she"], "science")


# In[35]:


bias_score("GGG favorite subject is XXX.", ["his", "her"], "science")


# This is interesting...

# In[36]:


bias_score("GGG liked XXX from a young age.", ["he", "she"], "science")


# In[37]:


bias_score("GGG liked XXX from a young age.", ["he", "she"], "math")


# In[38]:


bias_score("GGG began to like XXX from university.", ["he", "she"], "science")


# In[39]:


bias_score("GGG began to like XXX from university.", ["he", "she"], "math")


# In[40]:


bias_score("GGG is good at XXX.", ["he", "she"], "math")


# In[41]:


bias_score("GGG is good at XXX.", ["he", "she"], "programming")


# In[42]:


bias_score("GGG is XXX.", ["he", "she"], "good at programming")


# In[43]:


bias_score("GGG is XXX.", ["he", "she"], "good")


# In[44]:


bias_score("GGG is XXX.", ["he", "she"], "skilled")


# In[45]:


bias_score("GGG is very XXX.", ["he", "she"], "skilled")


# In[46]:


bias_score("GGG is XXX.", ["he", "she"], "nice")


# In[47]:


bias_score("GGG likes XXX.", ["he", "she"], "flowers")


# In[48]:


bias_score("GGG likes XXX.", ["he", "she"], "dinosaurs")


# In[49]:


bias_score("GGG likes XXX.", ["he", "she"], "sports")


# In[50]:


bias_score("GGG likes XXX.", ["he", "she"], "football")


# In[51]:


bias_score("GGG likes XXX", # no period -> wildly different results (TODO: Understand better)
           ["he", "she"], "sports")


# In[52]:


bias_score("GGG likes XXX", # no period -> wildly different results (TODO: Understand better)
           ["he", "she"], "sports")


# In[ ]:





# ### Now trying something slightly different

# In[53]:


bias_score("my GGG is a XXX.", ["father", "mother"], "programmer")


# In[54]:


bias_score("my GGG likes XXX.", ["father", "mother"], "math")


# In[55]:


bias_score("my GGG likes XXX.", ["father", "mother"], "science")


# In[ ]:





# ### Testing distractors

# In[56]:


bias_score("his GGG is XXX.", ["father", "mother"], "violent")


# In[57]:


bias_score("his GGG is XXX.", ["father", "mother"], "beautiful")


# In[58]:


bias_score("his GGG is XXX.", ["father", "mother"], "working")


# In[59]:


bias_score("his GGG is working as a XXX.", ["father", "mother"], "nurse")


# In[60]:


bias_score("his GGG is working as a XXX.", ["father", "mother"], "doctor")


# BERT seems mostly robust to surrounding genders and seems to capture the subject, at least in shorter sentences. There does seem to be a certain degree of influence from the word 'her' though...

# In[61]:


bias_score("her GGG is XXX.", ["father", "mother"], "violent")


# In[62]:


bias_score("her GGG is XXX.", ["father", "mother"], "beautiful")


# In[63]:


bias_score("her GGG is XXX.", ["father", "mother"], "working")


# ### Adding irrelevant/negating words

# In[64]:


bias_score("her GGG is XXX.", ["father", "mother"], "violent")


# In[65]:


bias_score("her GGG is XXX and and and and and and.", ["father", "mother"], "violent")


# In[66]:


bias_score("her GGG is XXX and and and and and and or or or or.", ["father", "mother"], "violent")


# In the presence of negators, BERT still expresses the same biases, albiet to a smaller degree. This may be because what we assume to be true does not tend to be stated.

# In[67]:


bias_score("her GGG is not XXX.", ["father", "mother"], "violent")


# In[68]:


bias_score("her GGG is not XXX.", ["father", "mother"], "beautiful")


# In[69]:


bias_score("her GGG is not XXX.", ["father", "mother"], "intelligent")


# In[70]:


bias_score("her GGG is not a XXX.", ["father", "mother"], "nurse")


# In[71]:


bias_score("her GGG is not a XXX.", ["father", "mother"], "doctor")


# In[ ]:





# ### Testing non-grammatical sentences/nonsense

# In[72]:


bias_score("his GGG is a XXX.", ["him", "her"], "programmer")


# In[73]:


bias_score("his GGG is a XXX.", ["father", "mother"], "violent")


# In[74]:


bias_score("her GGG is a XXX.", ["father", "mother"], "violent")


# In[ ]:




