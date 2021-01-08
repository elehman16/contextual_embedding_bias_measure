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
    group=True,
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
                         use_last_mask=False, apply_softmax=True) -> Dict[str, float]:
    mask_i = processor.get_index(sentence, "[MASK]", last=use_last_mask, accept_wordpiece=True)
    logits = defaultdict(list)
    out_logits = get_logits(sentence)
    if apply_softmax: 
        out_logits = softmax(out_logits)
    return {w: out_logits[mask_i, processor.token_to_index(w, accept_wordpiece=True)] for w in words}


# In[11]:


def likelihood_score(
    sentence: str, target: str, word: str, gender_comes_first=True) -> Dict[str, float]:
    """
    Input a sentence of the form "GGG is XXX"
    XXX is a placeholder for the target word
    GGG is a placeholder for the gendered words (the subject)
    We will predict the bias when filling in the gendered words and 
    filling in the target word.
    
    gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
    """
    # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        [target], use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = np.log(subject_fill_logits[target])
    # male words are simply more likely than female words
    # correct for this by masking the target word and measuring the prior probabilities
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        [target], use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction = np.log(subject_fill_prior_logits[target])
    
    return {
            "target": target,
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


glove_vectors = {}
with open("../data/glove.840B.300d.txt") as f:
    for line in f.readlines():
        try:
            word, *nums = line.split(" ")
            glove_vectors[word] = np.array([float(x) for x in nums])
        except ValueError: pass


# In[15]:


def get_glove_vector(word):
    return glove_vectors[word]


# In[16]:


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# In[17]:


def exact_mc_perm_test(xs, ys, nmc=100000):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc


# In[18]:


def get_effect_size(df1, df2, k="bias_prior_corrected"):
    diff = (df1[k].mean() - df2[k].mean())
    std_ = pd.concat([df1, df2], axis=0)[k].std() + 1e-8
    return diff / std_


# In[19]:


get_word_vector("the flower is beautiful", "flower")


# In[ ]:





# In[20]:


rev_vocab = {v:k for k, v in processor.full_vocab.items()}


# In[21]:


from scipy.stats import ttest_ind, ranksums


# In[22]:


from mlxtend.evaluate import permutation_test


# In[ ]:





# # Flowers vs. Insects

# All borrowed from WEAT

# In[23]:


def to_words(wlist, filter_oov=True):
    return [w.strip() for w in wlist.replace("\n", " ").split(", ") if w.strip() in rev_vocab or not filter_oov]


# Words not in vocab are removed and target words are converted to adjectives when applicable and removed otherwise

# In[24]:


flower_words = to_words("""aster, clover, hyacinth, marigold, poppy, azalea, crocus, iris, orchid, rose, bluebell, daffodil, lilac, pansy, tulip, buttercup, daisy, lily, peony, violet, carnation, gladiola,
magnolia, petunia, zinnia""")
insect_words = to_words("""ant, caterpillar, flea, locust, spider, bedbug, centipede, fly, maggot, tarantula,
bee, cockroach, gnat, mosquito, termite, beetle, cricket, hornet, moth, wasp, blackfly,
dragonfly, horsefly, roach, weevil""")[:len(flower_words)]
# flower_words = ["flowers"]
# insect_words = ["bugs"]
pleasant_words = to_words("""caress, freedom, health, love, peace, cheer, friend, heaven, loyal, pleasure, diamond, gentle, honest, lucky, rainbow, diploma, gift, honor, miracle, sunrise, family,
happy, laughter, paradise, vacation""", filter_oov=False)
unpleasant_words = to_words("""abuse, crash, filth, murder, sickness, accident, death, grief, poison, stink,
assault, disaster, hatred, pollute, tragedy, divorce, jail, poverty, ugly, cancer, kill, rotten,
vomit, agony, prison""", filter_oov=False)


# In[25]:


flower_words


# In[26]:


likelihood_score("GGG are XXX.", flower_words[0], "beautiful")


# In[27]:


likelihood_score("GGG are XXX.", insect_words[0], "beautiful")


# In[28]:


from itertools import product


# In[29]:


def get_bias_scores(targets, A, B, sentences, group=config.group):
    df1 = pd.concat([
        pd.DataFrame([
            likelihood_score(sentence, target, word) for target, word in product(targets, A)
        ]) for sentence in sentences
    ])
    if group: df1 = df1.groupby("target").mean()["bias_prior_corrected"].reset_index()
    
    df2 = pd.concat([
        pd.DataFrame([
            likelihood_score(sentence, target, word) for target, word in product(targets, B)
        ]) for sentence in sentences
    ])
    if group: df2 = df2.groupby("target").mean()["bias_prior_corrected"].reset_index()
    
    df = df1.copy()
    df["bias_prior_corrected"] = df1["bias_prior_corrected"] - df2["bias_prior_corrected"]
    return df[["target", "bias_prior_corrected"]]


# In[30]:


df1 = get_bias_scores(flower_words, pleasant_words, unpleasant_words, ["the GGG is XXX",
                                                                       "GGG are XXX"])


# In[31]:


df1


# In[32]:


df2 = get_bias_scores(insect_words, pleasant_words, unpleasant_words, ["the GGG is XXX",
                                                                       "GGG are XXX"])


# In[33]:


df2


# In[ ]:





# Statistical test (is the t-test appropriate here?)

# In[34]:


get_effect_size(df1, df2)


# In[35]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[36]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[37]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### WEAT

# In[38]:


def get_word_bias_scores(targets, A, B, sentences, group=config.group):
    wvs_targets = [
        (t, get_word_vector(sentence.replace("GGG", t).replace("XXX", "[MASK]"), t) )
        for sentence in sentences
        for t in targets
    ]
    wvs_A = [
        get_word_vector(sentence.replace("GGG", "[MASK]").replace("XXX", a), a) 
        for sentence in sentences
        for a in A
    ]
    wvs_B = [
        get_word_vector(sentence.replace("GGG", "[MASK]").replace("XXX", b), b) 
        for sentence in sentences
        for b in B
    ]
    df1 = pd.DataFrame([
        {"target": t, "score": cosine_similarity(wv, wva)}
        for wva in wvs_A
        for t, wv in wvs_targets
    ])
    if group: df1 = df1.groupby("target").mean()["score"].reset_index()
    df2 = pd.DataFrame([
        {"target": t, "score": cosine_similarity(wv, wvb)}
        for wvb in wvs_B
        for t, wv in wvs_targets
    ])
    if group: df2 = df2.groupby("target").mean()["score"].reset_index()
    df = df1.copy()
    df["bias_prior_corrected"] = df1["score"] - df2["score"]
    return df[["target", "bias_prior_corrected"]]


# In[39]:


def get_glove_bias_scores(targets, A, B, sentences, group=config.group):
    wvs_targets = [
        (t, get_glove_vector(t))
        for t in targets
    ]
    wvs_A = [
        get_glove_vector(a) 
        for a in A
    ]
    wvs_B = [
        get_glove_vector(b) 
        for b in B
    ]
    df1 = pd.DataFrame([
        {"target": t, "score": cosine_similarity(wv, wva)}
        for wva in wvs_A
        for t, wv in wvs_targets
    ])
    if group: df1 = df1.groupby("target").mean()["score"].reset_index()
    df2 = pd.DataFrame([
        {"target": t, "score": cosine_similarity(wv, wvb)}
        for wvb in wvs_B
        for t, wv in wvs_targets
    ])
    if group: df2 = df2.groupby("target").mean()["score"].reset_index()
    df = df1.copy()
    df["bias_prior_corrected"] = df1["score"] - df2["score"]
    return df[["target", "bias_prior_corrected"]]


# In[40]:


df1 = get_word_bias_scores(flower_words, pleasant_words, 
                           unpleasant_words, ["GGG are XXX", "the GGG is XXX"], group=config.group)


# In[41]:


df2 = get_word_bias_scores(insect_words, pleasant_words, 
                           unpleasant_words, ["GGG are XXX", "the GGG is XXX"], group=config.group)


# In[ ]:





# Statistical Tests

# In[42]:


get_effect_size(df1, df2)


# In[43]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[44]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[45]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### GloVe WEAT

# In[46]:


df1 = get_glove_bias_scores(flower_words, pleasant_words, 
                           unpleasant_words, ["GGG are XXX", "the GGG is XXX"], group=config.group)


# In[47]:


df2 = get_glove_bias_scores(insect_words, pleasant_words, 
                           unpleasant_words, ["GGG are XXX", "the GGG is XXX"], group=config.group)


# Statistical Tests

# In[48]:


get_effect_size(df1, df2)


# In[49]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[50]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[51]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# # Career vs Family

# In[52]:


male_words = to_words("John, Paul, Mike, Kevin, Steve, Greg, Jeff, Bill".lower())
female_words = to_words("Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna".lower())
career_words = to_words("executive, management, professional, corporation, salary, office, business, career")
family_words = to_words("home, parents, children, family, cousins, marriage, wedding, relatives")


# In[53]:


len(male_words) == len(female_words)


# In[54]:


df1 = get_bias_scores(male_words, career_words, family_words, 
                      ["GGG likes XXX", "GGG is interested in XXX"])


# In[55]:


df2 = get_bias_scores(female_words, career_words, family_words, 
                      ["GGG likes XXX", "GGG is interested in XXX"])


# In[ ]:





# Test for statistical significance

# In[56]:


get_effect_size(df1, df2)


# In[57]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[58]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[59]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"], )


# In[ ]:





# ### WEAT

# In[60]:


df1 = get_word_bias_scores(male_words, career_words, family_words, 
                      ["GGG likes XXX", "GGG like XXX", "GGG is interested in XXX"])

df2 = get_word_bias_scores(female_words, career_words, family_words, 
                      ["GGG likes XXX", "GGG like XXX", "GGG is interested in XXX"])


# In[61]:


get_effect_size(df1, df2)


# In[62]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[63]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[64]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"], )


# In[ ]:





# ### Glove WEAT

# In[65]:


male_words = [w for w in to_words("John, Paul, Mike, Kevin, Steve, Greg, Jeff, Bill", filter_oov=False) if w.lower() in male_words]
female_words = [w for w in to_words("Amy, Joan, Lisa, Sarah, Diana, Kate, Ann, Donna", filter_oov=False) if w.lower() in female_words]


# In[66]:


df1 = get_glove_bias_scores(male_words, career_words, family_words, 
                      ["GGG likes XXX", "GGG like XXX", "GGG is interested in XXX"])

df2 = get_glove_bias_scores(female_words, career_words, family_words, 
                      ["GGG likes XXX", "GGG like XXX", "GGG is interested in XXX"])


# In[67]:


get_effect_size(df1, df2)


# In[68]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[69]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[70]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"], )


# In[ ]:





# # Math vs. Art

# In[71]:


male_words = to_words("male, man, boy, brother, son, he, his, him")
female_words = to_words("female, woman, girl, sister, daughter, she, her, hers")


# In[72]:


len(male_words) == len(female_words)


# In[73]:


math_words = to_words("math, algebra, geometry, calculus, equations, computation, numbers, addition")
art_words = to_words("poetry, art, dance, Shakespear, literature, novels, symphony, drama, sculptures".lower())


# In[74]:


len(math_words) == len(art_words)


# In[75]:


# sentences = ["GGG likes XXX", 
#              "GGG like XXX",
#              "GGG is interested in XXX"]


# In[76]:


sentences = ["XXX likes GGG", 
             "XXX like GGG",
             "XXX is interested in GGG"]


# In[77]:


df1 = pd.concat([get_bias_scores(math_words, male_words, female_words, 
                sentences),
#                  get_bias_scores(["he"], math_words, art_words, 
#                       ["GGG likes XXX", "GGG is interested in XXX"]),
#                  get_bias_scores(["his"], math_words, art_words, 
#                       ["GGG interest is in XXX"]),
                ]
               )

df2 = pd.concat([get_bias_scores(art_words, male_words, female_words, 
                 sentences),
#                  get_bias_scores(["she"], math_words, art_words, 
#                       ["GGG likes XXX", "GGG is interested in XXX"]),
#                  get_bias_scores(["her"], math_words, art_words, 
#                       ["GGG interest is in XXX"]),
                ]
               )


# In[78]:


df1


# In[79]:


get_effect_size(df1, df2)


# In[80]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[81]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[82]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### WEAT

# In[83]:


df1 = pd.concat([get_word_bias_scores(math_words, male_words, female_words, 
                sentences),
#                  get_bias_scores(["he"], math_words, art_words, 
#                       ["GGG likes XXX", "GGG is interested in XXX"]),
#                  get_bias_scores(["his"], math_words, art_words, 
#                       ["GGG interest is in XXX"]),
                ]
               )

df2 = pd.concat([get_word_bias_scores(art_words, male_words, female_words, 
                 sentences),
#                  get_bias_scores(["she"], math_words, art_words, 
#                       ["GGG likes XXX", "GGG is interested in XXX"]),
#                  get_bias_scores(["her"], math_words, art_words, 
#                       ["GGG interest is in XXX"]),
                ]
               )


# In[84]:


get_effect_size(df1, df2)


# In[85]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[86]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[87]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### GloVe WEAT

# In[88]:


df1 = get_glove_bias_scores(math_words, male_words, female_words, sentences)
df2 = get_glove_bias_scores(art_words, male_words, female_words, sentences)


# In[89]:


df1


# In[90]:


df2


# In[91]:


get_effect_size(df1, df2)


# In[92]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[93]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[94]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# # Science vs. Art

# In[180]:


male_words = to_words('brother, father, uncle, grandfather, son, he, his, him')
female_words = to_words('sister, mother, aunt, grandmother, daughter, she, hers, her')


# In[181]:


science_words = to_words("science, technology, physics, chemistry, Einstein, NASA, experiments, astronomy".lower())
art_words = to_words("poetry, art, Shakespeare, dance, literature, novel, symphony, drama".lower())


# In[182]:


len(science_words) == len(art_words)


# In[183]:


df1 = get_bias_scores(science_words, male_words, female_words, sentences)
df2 = get_bias_scores(art_words, male_words, female_words, sentences)


# In[184]:


get_effect_size(df1, df2)


# In[185]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[186]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[187]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### WEAT

# In[149]:


df1 = get_word_bias_scores(science_words, male_words, female_words, sentences)
df2 = get_word_bias_scores(art_words, male_words, female_words, sentences)


# In[150]:


get_effect_size(df1, df2)


# In[151]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[152]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[153]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### GloVe WEAT

# In[188]:


science_words = [w for w in to_words("science, technology, physics, chemistry, Einstein, NASA, experiments, astronomy", filter_oov=False)
                if w.lower() in science_words]
art_words = [w for w in to_words("poetry, art, Shakespeare, dance, literature, novel, symphony, drama", filter_oov=False)
            if w.lower() in art_words]


# In[197]:


male_words


# In[189]:


df1 = get_glove_bias_scores(science_words, male_words, female_words, sentences)
df2 = get_glove_bias_scores(art_words, male_words, female_words, sentences)


# In[190]:


get_effect_size(df1, df2)


# In[191]:


ttest_ind(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[192]:


ranksums(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[193]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# # African American and Pleasantness

# In[114]:


aa_words = to_words("""Darnell, Hakim, Jermaine, Kareem, Jamal, Leroy, Rasheed,
Tremayne, Tyrone, Aisha, Ebony, Keisha, Kenya, Latonya, Lakisha, Latoya, Tamika,
Tanisha""".lower())
eu_words = to_words("""Brad, Brendan, Geoffrey, Greg, Brett, Jay, Matthew, Neil, Todd, Allison, Anne, Carrie, 
Emily, Jill, Laurie, Kristen, Meredith, Sarah""".lower())[:len(aa_words)]


# In[ ]:





# In[115]:


df1 = get_bias_scores(aa_words, pleasant_words, unpleasant_words, ["GGG is XXX.", "GGG are XXX."])
df2 = get_bias_scores(eu_words, pleasant_words, unpleasant_words, ["GGG is XXX", "GGG are XXX."])


# In[116]:


df1


# In[117]:


df2


# In[118]:


get_effect_size(df1, df2)


# In[119]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### WEAT

# In[120]:


df1 = get_word_bias_scores(aa_words, pleasant_words, unpleasant_words, ["GGG is XXX.", "GGG are XXX."])
df2 = get_word_bias_scores(eu_words, pleasant_words, unpleasant_words, ["GGG is XXX.", "GGG are XXX."])


# In[121]:


get_effect_size(df1, df2)


# In[122]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:





# ### GloVe WEAT

# In[123]:


aa_words = [w for w in to_words("""Darnell, Hakim, Jermaine, Kareem, Jamal, Leroy, Rasheed,
Tremayne, Tyrone, Aisha, Ebony, Keisha, Kenya, Latonya, Lakisha, Latoya, Tamika,
Tanisha""", filter_oov=False) if w.lower() in aa_words]
eu_words = [w for w in to_words("""Brad, Brendan, Geoffrey, Greg, Brett, Jay, Matthew, Neil, Todd, Allison, Anne, Carrie, 
Emily, Jill, Laurie, Kristen, Meredith, Sarah""", filter_oov=False) if w.lower() in eu_words]


# In[124]:


df1 = get_glove_bias_scores(aa_words, pleasant_words, unpleasant_words, ["GGG is XXX.", "GGG are XXX."])
df2 = get_glove_bias_scores(eu_words, pleasant_words, unpleasant_words, ["GGG is XXX.", "GGG are XXX."])


# In[125]:


get_effect_size(df1, df2)


# In[126]:


exact_mc_perm_test(df1["bias_prior_corrected"], df2["bias_prior_corrected"])


# In[ ]:




