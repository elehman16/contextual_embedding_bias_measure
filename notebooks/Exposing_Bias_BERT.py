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
import sys
sys.path.append("../lib")
from bert_utils import Config, BertPreprocessor


# In[3]:


from bert_expose_bias_with_prior import *
from construct_bias_score import *


# In[13]:


def Txt2List(file):
    ll=[]
    with open(file) as f:
        for line in f:
            ll.append(line.strip().lower())
    return ll
        
def plot_pie(file, mc=50, fc=50):
#mc=50
#fc=50
    # Data to plot
    labels = 'Male', 'Female'
    sizes = [mc, fc]
    colors = ['lightcoral', 'lightskyblue']

    # Plot
    fig = plt.figure()
    plt.pie(sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    fig.savefig(file)
    plt.show()
    
def list2Bias_norm(plotfile, var_list, abs_str, print_str): #Example of abs_str is "good at ", print_str is "is good at "
    mc=0
    fc=0
    for var in var_list:
        strr = abs_str+ var
        ans = bias_score("GGG is XXX", ["he", "she"], strr)
        score= ans['gender_fill_bias_prior_corrected']

        if score>=0:
            mc+=1
            print("Man ",print_str,  var, " by ", score)

        else:
            fc+=1
            print("Woman ",print_str,  var, " by ", score)



    plot_pie(plotfile, mc, fc)
    
    
def list2Bias(plotfile, var_list, abs_str): #Example of abs_str "is good at "
    mc=0
    fc=0
    for var in var_list:
        
        score = get_log_odds("[MASK] %s%s"%(abs_str,var), "he", "she")
        
        if score>=0:
            mc+=1
            print("Man ",abs_str,  var, " by ", score)

        else:
            fc+=1
            print("Woman ",abs_str,  var, " by ", score)
        

    plot_pie(plotfile, mc, fc)
    
    
    


# # Exposing Bias in BERT
# 
# 
# In this notebook, I'll experiment with a couple of possibilities for exposing Bias in BERT. We will concentrate on gender bias for now and look at a possible extension for racial Bias.
# 
# 
# I am trying to look for ways of exposing bias that have a clear negative impact on the party against which the bias is present. But for each of these, we will need a good dataset (so, we might want to replace current datasets with larger/reliable/authoritative datasets in the future)
# 
# 
# Currently, I am using the Masked Prediction Task but we might be able to extend this for Next Sentence Prediction as well.
# 
# We will classify the types of negative impact that are possible and look at experiments on their possible causes-
# 
# 
# 
# ## 1) Economic/Professional Impact-
# 
# Employers may use searching/ranking based on certain skills or job titles . They might want to specifically look for certain skills, traits and impactful positions.
# 
# 
# ### a) Bias for associating advanced skills with a  group - 
# 
# Example-  "Cloud Computing", "Machine Learning" , "Deep Learning" , "Management" etc.
# 
# #### Dataset: https://learning.linkedin.com/blog/top-skills/the-skills-companies-need-most-in-2019--and-how-to-learn-them
# 
# #### 25X In-Demand Tech Skills according to Linkedin
# 
# 
# 
# 

# In[9]:


#Load Dataset
skills = Txt2List('data/in_demand_tech_skills')


# In[10]:


# Before Removing Prior

list2Bias('in_demand_tech_skills.pdf', skills, "is good at ")


# In[15]:


# After Removing Prior

list2Bias_norm('in_demand_tech_skills_without_prior.pdf', skills, "good at ", "is good at ")


# ### b) Bias for associating positive traits with a group - 
# 
# #### Dataset- http://ideonomy.mit.edu/essays/traits.html

# In[18]:


#Load Dataset
pos_traits_list = Txt2List('data/positive_traits')


# In[20]:


list2Bias('positive_traits.pdf', pos_traits_list, "is ")


# In[21]:


list2Bias_norm('positive_traits_without_prior.pdf', pos_traits_list, " ", "is ")


# ### c) Bias for associating negative traits with a  group - 

# In[23]:


#Load Dataset
neg_traits_list = Txt2List('data/negative_traits')


# In[24]:


list2Bias('negative_traits.pdf', neg_traits_list, "is ")


# In[25]:



list2Bias_norm('negative_traits_without_prior.pdf', neg_traits_list, " ", "is ")


# ### d) Bias for associating high salary jobs with a group - 
# 
# #### Dataset- https://catalog.data.gov/dataset/employee-salaries-2017

# In[26]:


#Load Dataset

Title=[]
Salary=[]
flag=0
def isFloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

with open('data/employeesalaries2017.csv') as f:
    for line in f:
        if flag==0:
            flag=1
            continue
        row = line.split(',')
        Title.append(row[2])
        if isFloat(row[8]):
            Salary.append(float(row[8]))
        else:
            Title.pop()
            
            
        
Title_sorted = sorted(Title,key=dict(zip(Title, Salary)).get,reverse=True)

unique_titles= set()

Top_Titles= []


for i in Title_sorted:
    if i in unique_titles:
        continue
    else:
        Top_Titles.append(i.lower())
        unique_titles.add(i)


# In[27]:



list2Bias('TopTitles.pdf', Top_Titles, "is ")


# In[29]:



list2Bias_norm('TopTitles_without_prior.pdf', Top_Titles, " ", "is ")


# ### e) Associating skills sought by Google with a group
# 
# #### Dataset: https://www.kaggle.com/niyamatalmass/google-job-skills

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### f) Associating skills sought by Amazon with a group
# 
# #### Dataset: https://www.kaggle.com/atahmasb/amazon-job-skills

# In[ ]:





# In[ ]:





# In[ ]:





# ### h) Associating skills sought by US based jobs (20k) on Dice.com
# 
# #### Dataset: https://www.kaggle.com/PromptCloudHQ/usbased-jobs-from-dicecom

# In[ ]:





# In[ ]:





# In[ ]:





# 
# # -----ROUGH------
# 
# 
# ### b) Bias for associating impactful roles with a group-
# 
# (Imperfect) Proxies for measuring impact- Salary, Prestige & Mixed (Based on dataset)
# 
# 2.1 Salary
# 
# #### Datasets:
# 
# https://www.careeronestop.org/Toolkit/Wages/highest-paying-careers.aspx
# 
# 2.2 Prestige: 
# 
# #### Datasets:
# 
# https://www.businessinsider.com/most-prestigious-jobs-in-america-2014-11
# 
# 
# 
# 
# 2.3 Other methodologies
# 
# Dataset: 
# 
# 
# 
# 
# 3. Bias for associating professional traits with a group-
# 
# #### Datasets:
# 
# https://www.monster.ca/career-advice/article/50-personality-traits-for-the-workplace-canada
# 
# https://learning.linkedin.com/blog/top-skills/the-skills-companies-need-most-in-2019--and-how-to-learn-them
# 
# 
# Social/Cultural Impact-
# 
# 1. Bias for associating negative traits (Eg- 'neurotic', 'weak') with a group
# 
# 2. Bias for associating certain life roles  (Eg- 'homemaker' , 'bread winner') with a group
# 
# 
# 
# 
# Some of this maybe extended to race as well-
# 
# 1. Bias for associating negative traits with a group
# 
# 2. Bias for associating profesional traits with a group
# 
# 

# In[ ]:




