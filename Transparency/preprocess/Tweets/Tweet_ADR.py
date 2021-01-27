#!/usr/bin/env python
# coding: utf-8

# ## Obtaining the datasets
# 
# For obtaining ADR tweets data, please contact the authors of 'Attention is Not Explanation' paper (https://arxiv.org/abs/1902.10186).  
# 

# In[13]:


import pandas as pd


# In[14]:


df = pd.read_csv('adr_dataset.csv')

from sklearn.model_selection import train_test_split
train_idx, dev_idx = train_test_split(df.index[df.exp_split == 'train'], test_size=0.15, random_state=16377)


# In[15]:


df.loc[dev_idx, 'exp_split'] = 'dev'
df.to_csv('adr_dataset_split.csv', index=False)


# In[16]:


get_ipython().run_line_magic('run', '"../preprocess_data_BC.py" --data_file adr_dataset_split.csv --output_file ./vec_adr.p --word_vectors_type fasttext.simple.300d --min_df 2')


# In[ ]:




