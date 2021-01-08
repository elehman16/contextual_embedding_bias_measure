#!/usr/bin/env python
# coding: utf-8

# Using the constrcuted dataset to test out the NIPS debiasing method

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
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import sys
sys.path.append("../lib")

DATA_ROOT = Path("../data")


# In[4]:


from bert_utils import Config, BertPreprocessor


# In[5]:


train_file = "gender_occ_pos_w_probs_train.txt"
val_file = "gender_occ_pos_w_probs_val.txt"


# In[6]:


config = Config(
    model_type="bert-base-uncased",
    max_seq_len=24,
    subspace_size=5,
)


# In[7]:


processor = BertPreprocessor(config.model_type, config.max_seq_len)


# In[8]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
model = BertForMaskedLM.from_pretrained(config.model_type)
model.eval()


# In[9]:


from dataclasses import dataclass

@dataclass
class ContextWord:
    sent: str
    word: str
    def __post_init__(self):
        assert self.word in self.sent


# In[10]:


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# In[11]:


def get_word_vector(cword: ContextWord, use_last_mask=False):
    sentence, word = cword.sent, cword.word
    idx = processor.get_index(sentence, word, last=use_last_mask)
    outputs = None
    with torch.no_grad():
        # TODO: Move to proper library function
        token_ids = processor.to_bert_model_input(sentence) 
        # ensure padding is consistent
        bert_input = torch.zeros(1, config.max_seq_len, dtype=torch.long)
        bert_input[0, :token_ids.size(1)] = token_ids
        sequence_output, _ = model.bert(bert_input,
                                        output_all_encoded_layers=False)
        sequence_output.squeeze_(0)
        if outputs is None: outputs = torch.zeros_like(sequence_output)
        outputs = sequence_output + outputs
    return outputs.detach().cpu().numpy()[idx]


# In[12]:


def construct_sim_matrix(vecs):
    sim_matrix = np.zeros((len(vecs), len(vecs)))
    for i, v in enumerate(vecs):
        for j, w in enumerate(vecs):
            sim_matrix[i, j] = cosine_similarity(v, w)
    return sim_matrix


# In[13]:


def construct_sim_matrix_df(sentences: List[str],
                           words: List[str]):
    sim = construct_sim_matrix([get_word_vector(ContextWord(sent, word)) for sent, word in zip(sentences, words)])
    return pd.DataFrame(data=sim, index=words, columns=words)


# In[14]:


def compute_diff_similarity(cwords1, cwords2):
    cword11, cword12 = cwords1
    cword21, cword22 = cwords2
    return cosine_similarity(get_word_vector(cword11) - get_word_vector(cword12),
                             get_word_vector(cword21) - get_word_vector(cword22))


# In[15]:


out_softmax = model.cls.predictions.decoder.weight.data.cpu().numpy()


# In[16]:


out_bias = model.cls.predictions.bias.data.cpu().numpy()


# In[17]:


def to_logits(wv: np.ndarray) -> np.ndarray:
    return model.cls(torch.FloatTensor(wv).unsqueeze(0)).detach().cpu().numpy()[0, :]


# In[ ]:





# # Check similarities

# In[18]:


construct_sim_matrix_df(["That person is a programmer.", 
                         "I am a man.", 
                         "I am a woman."],
                       ["programmer", "man", "woman"])


# In[19]:


compute_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The programmer went to the office.", "programmer"),
     ContextWord("The nurse went to the office.", "nurse"))
)


# In[20]:


compute_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The doctor went to the office.", "doctor"),
     ContextWord("The nurse went to the office.", "nurse"))
)


# In[21]:


compute_diff_similarity(
    (ContextWord("he likes sports.", "he"), ContextWord("she likes sports.", "she")),
    (ContextWord("The doctor went to the office.", "doctor"),
     ContextWord("The nurse went to the office.", "nurse"))
)


# In[ ]:





# # Find gendered direction

# In[22]:


df_train = pd.read_csv(DATA_ROOT / train_file)


# In[23]:


df_val = pd.read_csv(DATA_ROOT / val_file)


# In[24]:


from tqdm import tqdm

male_vecs, female_vecs = [], []
def add_word_vecs(s: str, male_w: str, female_w: str):
    male_vecs.append(get_word_vector(ContextWord(s.replace("[MASK]", male_w), male_w)))
    female_vecs.append(get_word_vector(ContextWord(s.replace("[MASK]", female_w), female_w)))

for i, row in tqdm(list(df_train.iterrows())):
    sentence = row["sentence"]
    add_word_vecs(sentence, row["mword"], row["fword"])


# In[25]:


male_vecs = np.r_[male_vecs]
female_vecs = np.r_[female_vecs]


# In[26]:


from sklearn.decomposition import PCA
def find_subspace(D: np.ndarray) -> PCA:
    assert len(D.shape) == 2
    pca = PCA(n_components=config.subspace_size)
    return pca.fit(D)


# In[27]:


pca = find_subspace(male_vecs - female_vecs)


# In[28]:


sns.barplot(x=np.arange(pca.n_components), y=pca.explained_variance_ratio_)


# In[ ]:





# This is what it says in the paper

# We denote the projection of a vector $ v $ onto $ B $ by

# $$ v_B = \sum_{j=1}^{k} (v \cdot b_j) b_j $$

# For each word $ w \in N $, let $ \vec{w} $ be re-embedded to
# $$ \vec{w} := \vec{w} - \vec{w_{B}} / || \vec{w} - \vec{w_{B}} || $$

# $$ \mu := \sum_{w \in E}w / |E| $$
# $$ \nu := \mu - \mu_B $$
# For each $ w \in E $, 
# $$ \vec{w} := \nu + \sqrt{1 - ||\nu||^2}\frac{\vec{w_B} - \mu_B}{||\vec{w_B} - \mu_B||} $$

# In[29]:


def remove_subspace(X: np.ndarray, subspace: np.ndarray, norm=True) -> np.ndarray:
    Xb = ((X @ subspace.T) @ subspace) # projection onto biased subspace
    X = (X - Xb) / (np.linalg.norm(X - Xb))
    if norm:
        mu = X.mean(0)
        mub = Xb.mean(0)
        nu = mu - mub
        return nu + np.sqrt(1 - nu**2) * (Xb - mub) / np.linalg.norm(Xb - mub)
    else:
        return X


# In[30]:


remove_subspace(male_vecs, pca.components_)


# In[ ]:





# ### Newly checking for differences

# In[31]:


def pp(X: np.ndarray) -> np.ndarray:
    """Postprocess"""
    return remove_subspace(np.expand_dims(X, 0), pca.components_, norm=False)[0]


# In[32]:


def compute_new_diff_similarity(cwords1, cwords2):
    cword11, cword12 = cwords1
    cword21, cword22 = cwords2
    return cosine_similarity(pp(get_word_vector(cword11)) - pp(get_word_vector(cword12)),
                             pp(get_word_vector(cword21)) - pp(get_word_vector(cword22)))


# Similarities are being reduced, so there is a shared gender subspace to a certain extent.

# In[33]:


(compute_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The programmer went to the office.", "programmer"),
     ContextWord("The nurse went to the office.", "nurse"))
),
compute_new_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The programmer went to the office.", "programmer"),
     ContextWord("The nurse went to the office.", "nurse"))
))


# In[34]:


(compute_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The doctor went to the office.", "doctor"),
     ContextWord("The nurse went to the office.", "nurse"))
),
compute_new_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The doctor went to the office.", "doctor"),
     ContextWord("The nurse went to the office.", "nurse"))
))


# In[35]:


(compute_diff_similarity(
    (ContextWord("he likes sports.", "he"), ContextWord("she likes sports.", "she")),
    (ContextWord("The doctor went to the office.", "doctor"),
     ContextWord("The nurse went to the office.", "nurse"))
),
compute_new_diff_similarity(
    (ContextWord("he likes sports.", "he"), ContextWord("she likes sports.", "she")),
    (ContextWord("The doctor went to the office.", "doctor"),
     ContextWord("The nurse went to the office.", "nurse"))
))


# In[ ]:





# # Checking for change in bias score

# Let's see if the bias score decreases with this transformation

# In[36]:


def compute_postprocess_bias_score(row):
    sentence, mword, fword, prior_bias = (row["sentence"], row["mword"], 
                                          row["fword"], row["prior_bias"])
    mwi, fwi = processor.token_to_index(mword), processor.token_to_index(fword)
    wv = get_word_vector(
        ContextWord(sentence, "[MASK]"),
        use_last_mask=True,
    )
    wv = pp(wv)
    logits = to_logits(wv)
    subject_fill_bias = logits[fwi] - logits[mwi]
    return subject_fill_bias - prior_bias


# In[ ]:





# Bias is reduced here

# In[37]:


tqdm.pandas()
df_train["bias_score_after"] = df_train.progress_apply(compute_postprocess_bias_score, axis=1)


# In[38]:


plt.hist(df_train["original_bias_score"])


# In[39]:


df_train["original_bias_score"].abs().mean()


# In[40]:


df_train["bias_score_after"].abs().mean()


# The bias score does not seem to be evenly reduced

# In[41]:


plt.hist(df_train["bias_score_after"])


# In[42]:


df_train[df_train["bias_score_after"].abs() > df_train["original_bias_score"].abs()]


# In[ ]:





# Evaluation on the validation set

# In[43]:


df_val["bias_score_after"] = df_val.progress_apply(compute_postprocess_bias_score, axis=1)


# In[44]:


plt.hist(df_val["original_bias_score"])


# In[45]:


plt.hist(df_val["bias_score_after"])


# In[46]:


df_val["original_bias_score"].abs().mean()


# In[47]:


df_val["bias_score_after"].abs().mean()


# In[ ]:





# # Unintended Side Effects

# Are there any unintended side effects of this transformation? Let's test and see

# In[48]:


compute_diff_similarity(
    (ContextWord("I am a man.", "man"), ContextWord("I am a woman.", "woman")),
    (ContextWord("The programmer went to the office.", "programmer"),
     ContextWord("The doctor went to the office.", "doctor"))
)


# In[49]:


def construct_sim_matrix_df(cws: List[ContextWord]):
   return pd.DataFrame(data=sim, index=words, columns=words)


# Before processing:

# In[50]:


cws = [
    ContextWord("The programmer went to the office.", "programmer"),
    ContextWord("The doctor went to the office.", "doctor"),
    ContextWord("The nurse went to the office.", "nurse"),
]
sim = construct_sim_matrix([get_word_vector(cw) for cw in cws])
pd.DataFrame(data=sim, index=[cw.word for cw in cws], columns=[cw.word for cw in cws])


# After processing:

# In[51]:


sim = construct_sim_matrix([pp(get_word_vector(cw)) for cw in cws])
pd.DataFrame(data=sim, index=[cw.word for cw in cws], columns=[cw.word for cw in cws])


# Interestingly, the similarities here seem to be roughly preserved; perhaps because we are neutralizing w.r.t to the gender dimension in the subject space, but not the object space?

# In[ ]:





# In[52]:


cws = [
    ContextWord("Your colleague is very beautiful.", "beautiful"),
    ContextWord("Your colleague is very dangerous.", "dangerous"),
    ContextWord("Your colleague is very normal.", "normal"),
]
sim = construct_sim_matrix([get_word_vector(cw) for cw in cws])
pd.DataFrame(data=sim, index=[cw.word for cw in cws], columns=[cw.word for cw in cws])


# Again, not much reduction in similarities here...

# In[53]:


sim = construct_sim_matrix([pp(get_word_vector(cw)) for cw in cws])
pd.DataFrame(data=sim, index=[cw.word for cw in cws], columns=[cw.word for cw in cws])


# In[ ]:




