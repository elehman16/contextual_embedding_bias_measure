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


# ### Prepare model

# In[7]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
model = BertForMaskedLM.from_pretrained(config.model_type)
model.eval()


# In[8]:


sequence_output, pooled_output = model.bert(processor.to_bert_model_input("hello world"),
                                            output_all_encoded_layers=False)


# In[9]:


def get_word_vector(sentence: str, word: str, n_calc: int=10):
    idx = processor.get_index(sentence, word)
    outputs = None
    with torch.no_grad():
        for _ in range(n_calc):
            sequence_output, _ = model.bert(processor.to_bert_model_input(sentence),
                                            output_all_encoded_layers=False)
            sequence_output.squeeze_(0)
            if outputs is None: outputs = torch.zeros_like(sequence_output)
            outputs = sequence_output + outputs
    return outputs.detach().cpu().numpy()[idx] / n_calc


# In[10]:


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# In[11]:


vec1 = get_word_vector("he is a programmer.", "programmer")


# In[12]:


vec2 = get_word_vector("she is a programmer.", "programmer")


# In[13]:


np.linalg.norm(vec1 - vec2)


# In[14]:


diff1 = vec1 - vec2


# In[15]:


out_softmax = model.cls.predictions.decoder.weight.data.cpu().numpy()


# In[16]:


out_softmax.shape


# In[17]:


ordering = ((out_softmax @ vec1)).argsort()


# In[18]:


(out_softmax @ vec1).shape


# In[19]:


processor.token_to_index("programmer")


# In[20]:


ordering = (-(out_softmax @ vec1)).argsort()


# In[21]:


(out_softmax @ vec1)[(out_softmax @ vec1).argsort()]


# In[ ]:





# In[22]:


ordering = (-(out_softmax @ vec1)).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[23]:


ordering = (-(out_softmax @ vec2)).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[24]:


ordering = (-(out_softmax @ (vec1 - vec2))).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[ ]:





# In[ ]:





# In[25]:


vec1 = get_word_vector("he is a person", "person")


# In[26]:


vec2 = get_word_vector("she is a person", "person")


# In[27]:


np.linalg.norm(vec1 - vec2)


# In[28]:


diff2 = vec1 - vec2


# In[ ]:





# In[29]:


np.dot(diff1, diff2) / (np.linalg.norm(diff1) * np.linalg.norm(diff2))


# In[ ]:





# In[30]:


vec1 = get_word_vector("he is [MASK].", "[MASK]")
vec2 = get_word_vector("she is [MASK].", "[MASK]")
diff3 = vec1 - vec2


# In[31]:


ordering = (-(out_softmax @ vec1)).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[32]:


ordering = (-(out_softmax @ vec2)).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[33]:


ordering = (-(out_softmax @ (vec1 - vec2))).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[ ]:





# ## Softmax layer analysis

# In[34]:


processor.full_vocab


# In[35]:


word_vectors ={
    word: out_softmax[i, :]
    for i, word in processor.full_vocab.items()
}


# In[36]:


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

from heapq import heappush, heappop
def nearest_neighbors(x, n=10):
    if isinstance(x, str):
        x = word_vectors[x]
    heap = []
    for w, v in word_vectors.items():
        sim = cosine_similarity(x, v)
        if len(heap) < n:
            heappush(heap, (sim, w))
        else:
            if heap[0] < (sim, w):
                heappop(heap)
                heappush(heap, (sim, w))
    return sorted(heap, reverse=True)


# In[37]:


nearest_neighbors("hello")


# In[38]:


nearest_neighbors("programmer")


# In[39]:


nearest_neighbors("doctor")


# In[40]:


nearest_neighbors("queen")


# In[41]:


nearest_neighbors(word_vectors["man"] - word_vectors["woman"] + word_vectors["king"])


# In[42]:


nearest_neighbors(word_vectors["man"])


# In[ ]:





# In[43]:


vec1 = get_word_vector("he is [MASK].", "[MASK]")
vec2 = get_word_vector("she is [MASK].", "[MASK]")


# In[44]:


diff = (-(out_softmax @ (vec1 - vec2)))
ordering = diff.argsort()
{
    diff[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[ ]:





# In[45]:


vec1 = get_word_vector("he is [MASK].", "is")
vec2 = get_word_vector("she is [MASK].", "is")
np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# In[46]:


vec3 = get_word_vector("he is [MASK].", "[MASK]")
vec4 = get_word_vector("she is [MASK].", "[MASK]")
np.dot(vec3, vec4) / (np.linalg.norm(vec3) * np.linalg.norm(vec4))


# In[47]:


cosine_similarity((vec1 - vec2), (vec3 - vec4))


# In[ ]:





# In[48]:


vec1 = get_word_vector("she is [MASK].", "[MASK]")
aaaa = out_softmax @ vec1
ordering = (-aaaa).argsort()
{
    aaaa[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[49]:


vec1 = get_word_vector("he is [MASK].", "[MASK]")
ordering = (-(out_softmax @ vec1)).argsort()
{
    ordering[i] + 1: processor.index_to_token(i)
    for i in ordering[:10]
}


# In[ ]:





# In[ ]:





# In[ ]:





# ### Comparing words in context

# In[50]:


vec1 = get_word_vector("he is a programmer.", "programmer")
vec2 = get_word_vector("she is a programmer.", "programmer")
vec3 = get_word_vector("the programmer wrote code on the board.", "programmer")


# In[51]:


cosine_similarity(vec1, vec2)


# In[52]:


cosine_similarity(vec1, vec3)


# In[ ]:





# In[53]:


vec1 = get_word_vector("he is a nurse.", "nurse")
vec2 = get_word_vector("she is a nurse.", "nurse")
vec3 = get_word_vector("the nurse wrote code on the board.", "nurse")


# In[54]:


cosine_similarity(vec1, vec2)


# In[55]:


cosine_similarity(vec1, vec3)


# In[ ]:





# In[56]:


vecs = []
vecs.append(get_word_vector("he is a programmer.", "programmer"))
vecs.append(get_word_vector("he is a programmer.", "he"))
vecs.append(get_word_vector("she is a programmer.", "programmer"))
vecs.append(get_word_vector("she is a programmer.", "she"))


# In[57]:


def construct_sim_matrix(vecs):
    sim_matrix = np.zeros((len(vecs), len(vecs)))
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            sim_matrix[i, j] = cosine_similarity(v, w)
    return sim_matrix


# In[58]:


construct_sim_matrix(vecs)


# In[ ]:





# In[59]:


vecs = []
vecs.append(get_word_vector("he is a programmer.", "he"))
vecs.append(get_word_vector("she is a programmer.", "she"))
vecs.append(get_word_vector("his profession is a programmer.", "his"))
vecs.append(get_word_vector("her profession is a programmer.", "her"))
vecs.append(get_word_vector("please talk to him.", "him"))
vecs.append(get_word_vector("please talk to her.", "her"))
vecs.append(get_word_vector("I work as a programmer.", "programmer"))
vecs.append(get_word_vector("I work as a nurse.", "nurse"))
vecs.append(get_word_vector("I work as a doctor.", "doctor"))
vecs.append(get_word_vector("I work as a nurse.", "nurse"))
vecs.append(get_word_vector("I am your father.", "father"))
vecs.append(get_word_vector("I am your mother.", "mother"))


# In[60]:


cosine_similarity(vecs[1]- vecs[0], vecs[3] - vecs[2])


# In[61]:


cosine_similarity(vecs[1]- vecs[0], vecs[5] - vecs[4])


# In[62]:


cosine_similarity(vecs[3]- vecs[2], vecs[5] - vecs[4])


# In[63]:


cosine_similarity(vecs[3]- vecs[2], vecs[7] - vecs[6])


# In[64]:


cosine_similarity(vecs[3]- vecs[2], vecs[9] - vecs[8])


# So, there does seem to be a gender subspace...?

# In[65]:


cosine_similarity(vecs[3]- vecs[2], vecs[11] - vecs[10])


# In[ ]:





# ### Checking for similarity

# Can't find much of a difference...

# In[66]:


prog_vec = get_word_vector("[MASK] is a programmer.", "programmer")
she_vec = get_word_vector("she is a programmer.", "she")
he_vec = get_word_vector("he is a programmer.", "he")
construct_sim_matrix([prog_vec, she_vec, he_vec])


# In[67]:


prog_vec = get_word_vector("[MASK] is a programmer.", "programmer")
she_vec = get_word_vector("she is a programmer.", "programmer")
he_vec = get_word_vector("he is a programmer.", "programmer")
construct_sim_matrix([prog_vec, she_vec, he_vec])


# In[ ]:





# ### Checking the distance between words in neutral contexts

# Programmer is slightly more similar to father than to mother

# In[124]:


def construct_sim_matrix_df(sentences: List[str],
                           words: List[str]):
    sim = construct_sim_matrix([get_word_vector(sent, word) for sent, word in zip(sentences, words)])
    return pd.DataFrame(data=sim, index=words, columns=words)


# In[138]:


construct_sim_matrix_df(["That person is a programmer.", "That person is my mother.", 
                         "That person is my father."],
                       ["programmer", "mother", "father"])


# Nurse is closer to mother

# In[128]:


construct_sim_matrix_df(["That person is a nurse.", "That person is my mother.", 
                         "That person is my father."],
                       ["nurse", "mother", "father"])


# In[132]:


construct_sim_matrix_df(["My nurse will not allow that.", "That person is my mother.", 
                         "That person is my father."],
                       ["nurse", "mother", "father"])


# Even the same word can have pretty different embeddings

# In[135]:


construct_sim_matrix_df(["Please don't let your mother eat that cookie.", "That person is my mother.", 
                         "That person is my father."],
                       ["mother", "mother", "father"])


# Different parts of speech lead to vastly different embeddings

# In[136]:


construct_sim_matrix_df(["The cat could mother that dog.", "That person is my mother.", 
                         "That person is my father."],
                       ["mother", "mother", "father"])


# Comparsions between different parts of speech: Still roughly the same pattern

# In[139]:


construct_sim_matrix_df(["That person is a nurse.", "What is she doing?", 
                         "What is he doing?"],
                       ["nurse", "she", "he"])


# In[ ]:





# In[ ]:





# # Gendered Subspace

# ### Construct gender subspace

# A simple test (TODO: Automate construction)

# In[68]:


male_vecs, female_vecs = [], []
def add_word_vecs(s: str, male_w: str, female_w: str):
    male_vecs.append(get_word_vector(s.replace("XXX", male_w), male_w))
    female_vecs.append(get_word_vector(s.replace("XXX", female_w), female_w))

for prof in ["musician", "magician", "nurse", "doctor", "teacher"]:
    add_word_vecs("XXX is a YYY".replace("YYY", prof), "he", "she")
    add_word_vecs("XXX works as a YYY".replace("YYY", prof), "he", "she")

for action in ["talk to", "hit", "ignore", "please", "remove"]:
    add_word_vecs("please YYY XXX".replace("YYY", action), "him", "her")
    add_word_vecs("don't YYY XXX".replace("YYY", action), "him", "her")

for thing in ["food", "music", "work", "running", "cooking"]:
    add_word_vecs("XXX likes YYY".replace("YYY", thing), "he", "she")
    add_word_vecs("XXX enjoys YYY".replace("YYY", thing), "he", "she")


# In[69]:


male_vecs = np.r_[male_vecs]
female_vecs = np.r_[female_vecs]


# In[70]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5)


# In[71]:


(male_vecs - female_vecs).shape


# In[95]:


diff_vecs = male_vecs - female_vecs


# In[97]:


X = svd.fit_transform(diff_vecs / (diff_vecs ** 2).sum(1, keepdims=True))


# In[113]:


svd.explained_variance_ratio_


# In[98]:


svd.explained_variance_ratio_.sum()


# In[99]:


svd.components_.shape


# In[ ]:





# ### Try eliminating this subspace and checking outputs softmax

# In[100]:


svd.components_


# In[101]:


vec = get_word_vector("[MASK] is a nurse.", "[MASK]")


# In[102]:


logits_before = (out_softmax @ vec)


# In[103]:


logits_before[processor.token_to_index("she")]


# In[104]:


logits_before[processor.token_to_index("he")]


# In[ ]:





# In[105]:


def eliminate_subspace(v, subspace):
    # TODO: Is there a better way?
    V = subspace
    beta = (np.linalg.inv(V @ V.T) @ V) @ v
    res = (v - (V.T @ beta))
    return res


# In[106]:


vec_after = eliminate_subspace(vec, svd.components_)


# In[107]:


logits_after = (out_softmax @ vec_after)


# The difference is indeed reduced

# In[108]:


logits_after[processor.token_to_index("she")]


# In[109]:


logits_after[processor.token_to_index("he")]


# In[ ]:





# Not quite working here...

# In[110]:


sentence = "[MASK] is a programmer."
vec = get_word_vector(sentence, "[MASK]")
logits_before = (out_softmax @ vec)
vec_after = eliminate_subspace(vec, svd.components_)
logits_after = (out_softmax @ vec_after)
print(f"Logit diff before: {logits_before[processor.token_to_index('she')] - logits_before[processor.token_to_index('he')]}")
print(f"Logit diff after: {logits_after[processor.token_to_index('she')] - logits_after[processor.token_to_index('he')]}")


# Hmm...

# In[111]:


sentence = "[MASK] is a housewife."
vec = get_word_vector(sentence, "[MASK]")
logits_before = (out_softmax @ vec)
vec_after = eliminate_subspace(vec, svd.components_)
logits_after = (out_softmax @ vec_after)
print(f"Logit diff before: {logits_before[processor.token_to_index('she')] - logits_before[processor.token_to_index('he')]}")
print(f"Logit diff after: {logits_after[processor.token_to_index('she')] - logits_after[processor.token_to_index('he')]}")


# Hmm...

# In[112]:


sentence = "[MASK] is my mother."
vec = get_word_vector(sentence, "[MASK]")
logits_before = (out_softmax @ vec)
vec_after = eliminate_subspace(vec, svd.components_)
logits_after = (out_softmax @ vec_after)
print(f"Logit diff before: {logits_before[processor.token_to_index('she')] - logits_before[processor.token_to_index('he')]}")
print(f"Logit diff after: {logits_after[processor.token_to_index('she')] - logits_after[processor.token_to_index('he')]}")


# In[ ]:





# In[ ]:




