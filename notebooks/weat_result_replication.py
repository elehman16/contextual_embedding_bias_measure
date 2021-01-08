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

def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)

from collections import defaultdict

def get_mask_fill_logits(sentence: str, words: Iterable[str],
                         use_last_mask=False, apply_softmax=True) -> Dict[str, float]:
    mask_i = processor.get_index(sentence, "[MASK]", last=use_last_mask, accept_wordpiece=True)
    logits = defaultdict(list)
    out_logits = get_logits(sentence)
    if apply_softmax: 
        out_logits = softmax(out_logits)
    return {w: out_logits[mask_i, processor.token_to_index(w, accept_wordpiece=True)] for w in words}

def bias_score(sentence: str, gender_words: Iterable[Iterable[str]], 
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
    mwords, fwords = gender_words
    all_words = mwords + fwords
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        all_words, use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = np.log(sum(subject_fill_logits[mw] for mw in mwords)) -                         np.log(sum(subject_fill_logits[fw] for fw in fwords))
    # male words are simply more likely than female words
    # correct for this by masking the target word and measuring the prior probabilities
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        all_words, use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction =             np.log(sum(subject_fill_prior_logits[mw] for mw in mwords)) -             np.log(sum(subject_fill_prior_logits[fw] for fw in fwords))
    
    return {
            "stimulus": word,
            "bias": subject_fill_bias,
            "prior_correction": subject_fill_bias_prior_correction,
            "bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
           }


# In[12]:


get_mask_fill_logits("the [MASK] is beautiful", ["flower", "bug"])


# In[13]:


def get_word_vector(sentence: str, word: str):
    idx = processor.get_index(sentence, word, accept_wordpiece=True)
    outputs = None
    with torch.no_grad():
        sequence_output, _ = model.bert(processor.to_bert_model_input(sentence),
                                        output_all_encoded_layers=False)
        sequence_output.squeeze_(0)
    return sequence_output.detach().cpu().numpy()[idx]


# In[14]:


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# In[15]:


def get_effect_size(df1, df2, k="bias_prior_corrected"):
    diff = (df1[k].mean() - df2[k].mean())
    std_ = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    return diff / std_


# In[16]:


def exact_mc_perm_test(xs, ys, nmc=100000):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc


# In[17]:


get_word_vector("the flower is beautiful", "flower")


# In[ ]:





# In[18]:


rev_vocab = {v:k for k, v in processor.full_vocab.items()}


# In[19]:


from scipy.stats import ttest_ind, ranksums


# In[20]:


from mlxtend.evaluate import permutation_test


# In[ ]:





# # Flowers vs. Insects

# All borrowed from WEAT

# In[21]:


def to_words(wlist, filter_oov=True):
    return [w.strip() for w in wlist.lower().replace("\n", " ").split(", ") if w.strip() in rev_vocab or not filter_oov]


# Words not in vocab are removed and target words are converted to adjectives when applicable and removed otherwise

# In[22]:


# flower_words = to_words("""aster, clover, hyacinth, marigold, poppy, azalea, crocus, iris, orchid, rose, bluebell, daffodil, lilac, pansy, tulip, buttercup, daisy, lily, peony, violet, carnation, gladiola,
# magnolia, petunia, zinnia""")
# insect_words = to_words("""ant, caterpillar, flea, locust, spider, bedbug, centipede, fly, maggot, tarantula,
# bee, cockroach, gnat, mosquito, termite, beetle, cricket, hornet, moth, wasp, blackfly,
# dragonfly, horsefly, roach, weevil""")
flower_single_words = ["flower"]
flower_words = ["flowers"]
insect_single_words = ["bug"]
insect_words = ["bugs"]
pleasant_words = to_words("""caress, freedom, health, love, peace, cheer, friend, heaven, loyal, pleasure, diamond, gentle, honest, lucky, rainbow, diploma, gift, honor, miracle, sunrise, family,
happy, laughter, paradise, vacation""", filter_oov=False)
unpleasant_words = to_words("""abuse, crash, filth, murder, sickness, accident, death, grief, poison, stink,
assault, disaster, hatred, pollute, tragedy, divorce, jail, poverty, ugly, cancer, kill, rotten,
vomit, agony, prison""", filter_oov=False)


# In[ ]:





# In[23]:


bias_score("GGG are XXX.", [flower_words, insect_words], "beautiful")


# In[24]:


bias_score("GGG are XXX.", [flower_words, insect_words], "pleasant")


# In[ ]:





# In[25]:


from itertools import product


# In[26]:


df1 = pd.concat([
pd.DataFrame([bias_score("the GGG is XXX.", 
                         [flower_words, insect_words], w) for w in pleasant_words]),
pd.DataFrame([bias_score("GGG are XXX.", 
                         [flower_single_words, insect_single_words], w) for w in pleasant_words]),
])
df1


# In[27]:


df1["bias_prior_corrected"].mean()


# In[28]:


df2 = pd.concat([
pd.DataFrame([bias_score("the GGG is XXX.", 
                         [flower_words, insect_words], w) for w in unpleasant_words]),
pd.DataFrame([bias_score("GGG are XXX.", 
                         [flower_single_words, insect_single_words], w) for w in unpleasant_words]),
])
df2


# In[29]:


df2["bias_prior_corrected"].mean()


# In[ ]:





# Statistical test (is the t-test appropriate here?)

# In[30]:


get_effect_size(df1, df2)


# In[31]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[32]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[33]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"], )


# In[ ]:





# ### WEAT

# In[34]:


wvs1 = [
    get_word_vector(f"[MASK] are {x}", x) for x in pleasant_words
]
wvs2 = [
    get_word_vector(f"[MASK] are {x}", x) for x in unpleasant_words
]


# In[35]:


wv_insect = get_word_vector("insects are [MASK]", "insects")
sims_insect1 = [cosine_similarity(wv_insect, wv) for wv in wvs1]
sims_insect2 = [cosine_similarity(wv_insect, wv) for wv in wvs2]
mean_diff = np.mean(sims_insect1) - np.mean(sims_insect2)
std_ = np.std(sims_insect1 + sims_insect2)
effect_sz_insect = mean_diff / std_; effect_sz_insect


# In[36]:


wv_flower = get_word_vector("flowers are [MASK]", "flowers")
sims_flower1 = [cosine_similarity(wv_flower, wv) for wv in wvs1]
sims_flower2 = [cosine_similarity(wv_flower, wv) for wv in wvs2]
mean_diff = np.mean(sims_flower1) - np.mean(sims_flower2)
std_ = np.std(sims_flower1 + sims_flower2)
effect_sz_flower = mean_diff / std_; effect_sz_flower


# In[37]:


exact_mc_perm_test(sims_insect1, sims_flower1)


# In[38]:


exact_mc_perm_test(sims_insect2, sims_flower2)


# In[ ]:





# In[ ]:





# # Career vs Family

# In[39]:


male_words = to_words("he")
female_words = to_words("she")
# male_words = to_words("John, Paul, Mike, Kevin, Steve, Greg, Jeff, Bill".lower())
# female_words = to_words("Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna".lower())
male_plural_words = to_words("boys, men")
female_plural_words = to_words("girls, women")
career_words = to_words("executive, management, professional, corporation, salary, office, business, career")
family_words = to_words("home, parents, children, family, cousins, marriage, wedding, relatives")


# In[40]:


df1 = pd.concat([
    pd.DataFrame([bias_score("GGG likes XXX.", [male_words, female_words], w) for w in career_words]),
    pd.DataFrame([bias_score("GGG like XXX.", [male_plural_words, female_plural_words], w) for w in career_words]),
    pd.DataFrame([bias_score("GGG is interested in XXX.", [["he"], ['she']], w) for w in career_words]), 
])
df1


# In[41]:


df1["bias_prior_corrected"].mean()


# In[42]:


df2 = pd.concat([
    pd.DataFrame([bias_score("GGG likes XXX.", [male_words, female_words], w) for w in family_words]),
    pd.DataFrame([bias_score("GGG like XXX.", [male_plural_words, female_plural_words], w) for w in family_words]),
    pd.DataFrame([bias_score("GGG is interested in XXX.", [["he"], ['she']], w) for w in family_words]), 
])
df2


# In[43]:


df2["bias_prior_corrected"].mean()


# In[ ]:





# In[ ]:





# Test for statistical significance

# In[44]:


get_effect_size(df1, df2)


# In[45]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[46]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[47]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"], )


# In[ ]:





# ### WEAT

# In[48]:


wvs1 = [
    get_word_vector(f"[MASK] like {x}", x) for x in family_words
] + [
    get_word_vector(f"[MASK] likes {x}", x) for x in family_words
] + [
    get_word_vector(f"[MASK] is interested in {x}", x) for x in family_words
]
wvs2 = [
    get_word_vector(f"[MASK] like {x}", x) for x in career_words
] + [
    get_word_vector(f"[MASK] likes {x}", x) for x in career_words    
] + [
    get_word_vector(f"[MASK] is interested in {x}", x) for x in career_words
]


# In[49]:


wv_fm = get_word_vector("women like [MASK]", "women")
wv_fm2 = get_word_vector("she likes [MASK]", "she")
sims_fm1 = [cosine_similarity(wv_fm, wv) for wv in wvs1] +           [cosine_similarity(wv_fm, wv) for wv in wvs1]
sims_fm2 = [cosine_similarity(wv_fm, wv) for wv in wvs2] +           [cosine_similarity(wv_fm2, wv) for wv in wvs2]
mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)
effect_sz_fm_family_career = mean_diff / std_; effect_sz_fm_family_career


# In[50]:


wv_m = get_word_vector("men like [MASK]", "men")
wv_m2 = get_word_vector("he likes [MASK]", "he")
sims_m1 = [cosine_similarity(wv_m, wv) for wv in wvs1]+           [cosine_similarity(wv_m2, wv) for wv in wvs1]
sims_m2 = [cosine_similarity(wv_m, wv) for wv in wvs2] +           [cosine_similarity(wv_m2, wv) for wv in wvs2]
mean_diff = np.mean(sims_m1) - np.mean(sims_m2)
std_ = np.std(sims_m1 + sims_m1)
effect_sz_m_family_career = mean_diff / std_; effect_sz_m_family_career


# In[51]:


exact_mc_perm_test(sims_fm1, sims_m1)


# In[52]:


exact_mc_perm_test(sims_fm2, sims_m2)


# In[ ]:





# # Math vs. Art

# In[53]:


math_words = to_words("math, algebra, geometry, calculus, equations, computation, numbers, addition")
art_words = to_words("poetry, art, dance, Shakespear, literature, novels, symphony, drama, sculptures")


# In[54]:


df1 = pd.concat([
    pd.DataFrame([bias_score("GGG likes XXX.", [male_words, female_words], w) for w in math_words]),
    pd.DataFrame([bias_score("GGG like XXX.", [male_plural_words, female_plural_words], w) for w in math_words]),
    pd.DataFrame([bias_score("GGG is interested in XXX.", [["he"], ['she']], w) for w in math_words]), 
])
df1


# In[55]:


df1["bias"].mean()


# In[56]:


df2 = pd.concat([
    pd.DataFrame([bias_score("GGG likes XXX.", [male_words, female_words], w) for w in art_words]),
    pd.DataFrame([bias_score("GGG like XXX.", [male_plural_words, female_plural_words], w) for w in art_words]),
    pd.DataFrame([bias_score("GGG is interested in XXX.", [["he"], ['she']], w) for w in art_words]),  
])
df2


# In[57]:


df2["bias"].mean()


# In[ ]:





# In[58]:


get_effect_size(df1, df2)


# In[59]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[60]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[61]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### WEAT

# In[62]:


wvs1 = [
    get_word_vector(f"[MASK] like {x}", x) for x in art_words
] + [
    get_word_vector(f"[MASK] likes {x}", x) for x in art_words
] + [
    get_word_vector(f"[MASK] is interested in {x}", x) for x in art_words
]
wvs2 = [
    get_word_vector(f"[MASK] like {x}", x) for x in math_words
] + [
    get_word_vector(f"[MASK] likes {x}", x) for x in math_words    
] + [
    get_word_vector(f"[MASK] is interested in {x}", x) for x in math_words
]


# In[63]:


sims_fm1 = [cosine_similarity(wv_fm, wv) for wv in wvs1] +           [cosine_similarity(wv_fm2, wv) for wv in wvs1]
sims_fm2 = [cosine_similarity(wv_fm, wv) for wv in wvs2] +           [cosine_similarity(wv_fm2, wv) for wv in wvs2]
mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)
effect_sz_fm_art_math = mean_diff / std_; effect_sz_fm_art_math

sims_m1 = [cosine_similarity(wv_m, wv) for wv in wvs1] +           [cosine_similarity(wv_m2, wv) for wv in wvs1]
sims_m2 = [cosine_similarity(wv_m, wv) for wv in wvs2] +           [cosine_similarity(wv_m2, wv) for wv in wvs2]
mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)
effect_sz_m_art_math = mean_diff / std_; effect_sz_m_art_math


# In[64]:


exact_mc_perm_test(sims_fm1, sims_m1)


# In[65]:


exact_mc_perm_test(sims_fm2, sims_m2)


# In[ ]:





# # Science vs. Art

# In[66]:


science_words = to_words("science, technology, physics, chemistry, Einstein, NASA, experiments, astronomy")
art_words = to_words("poetry, art, dance, Shakespear, literature, novels, symphony, drama, sculptures")


# In[67]:


df1 = pd.concat([
    pd.DataFrame([bias_score("GGG likes XXX.", [male_words, female_words], w) for w in science_words]),
    pd.DataFrame([bias_score("GGG like XXX.", [male_plural_words, female_plural_words], w) for w in science_words]),
    pd.DataFrame([bias_score("GGG is interested in XXX.", [["he"], ['she']], w) for w in science_words]), 
])
df1


# In[68]:


df1["bias"].mean()


# In[69]:


df2 = pd.concat([
    pd.DataFrame([bias_score("GGG likes XXX.", [male_words, female_words], w) for w in art_words]),
    pd.DataFrame([bias_score("GGG like XXX.", [male_plural_words, female_plural_words], w) for w in art_words]),
    pd.DataFrame([bias_score("GGG is interested in XXX.", [["he"], ['she']], w) for w in art_words]), 
])
df2


# In[70]:


df2["bias"].mean()


# In[71]:


get_effect_size(df1, df2)


# In[72]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[73]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[74]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### WEAT

# In[75]:


wvs1 = [
    get_word_vector(f"[MASK] like {x}", x) for x in art_words
] + [
    get_word_vector(f"[MASK] likes {x}", x) for x in art_words
] + [
    get_word_vector(f"[MASK] is interested in {x}", x) for x in art_words
]
wvs2 = [
    get_word_vector(f"[MASK] like {x}", x) for x in science_words
] + [
    get_word_vector(f"[MASK] likes {x}", x) for x in science_words    
] + [
    get_word_vector(f"[MASK] is interested in {x}", x) for x in science_words
]


# In[76]:


sims_fm1 = [cosine_similarity(wv_fm, wv) for wv in wvs1] +           [cosine_similarity(wv_fm2, wv) for wv in wvs1]
sims_fm2 = [cosine_similarity(wv_fm, wv) for wv in wvs2] +           [cosine_similarity(wv_fm2, wv) for wv in wvs2]
mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)
effect_sz_fm_art_math = mean_diff / std_; effect_sz_fm_art_math

sims_m1 = [cosine_similarity(wv_m, wv) for wv in wvs1] +           [cosine_similarity(wv_m2, wv) for wv in wvs1]
sims_m2 = [cosine_similarity(wv_m, wv) for wv in wvs2] +           [cosine_similarity(wv_m2, wv) for wv in wvs2]
mean_diff = np.mean(sims_fm1) - np.mean(sims_fm2)
std_ = np.std(sims_fm1 + sims_fm1)
effect_sz_m_art_math = mean_diff / std_; effect_sz_m_art_math


# In[77]:


exact_mc_perm_test(sims_fm1, sims_m1)


# In[78]:


exact_mc_perm_test(sims_fm2, sims_m2)


# In[ ]:





# # Black vs. White

# In[79]:


df1 = pd.concat([
pd.DataFrame([bias_score("GGG people are XXX.", 
                         [["black"], ["white"]], w) for w in pleasant_words]),
pd.DataFrame([bias_score("the GGG person is XXX.", 
                         [["black"], ["white"]], w) for w in pleasant_words]),])
df1


# In[80]:


df2 = pd.concat([
pd.DataFrame([bias_score("GGG people are XXX.", 
                         [["black"], ["white"]], w) for w in unpleasant_words]),
pd.DataFrame([bias_score("the GGG person is XXX.", 
                         [["black"], ["white"]], w) for w in unpleasant_words]),
])
df2


# In[81]:


get_effect_size(df1, df2)


# In[82]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# # Skills

# In[ ]:





# In[ ]:





# In[ ]:




