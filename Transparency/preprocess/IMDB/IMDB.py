#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget https://s3.amazonaws.com/text-datasets/imdb_full.pkl')
get_ipython().system('wget https://s3.amazonaws.com/text-datasets/imdb_word_index.json')


# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import pickle
data = pickle.load(open('imdb_full.pkl', 'rb'))

import json
vocab = json.load(open('imdb_word_index.json'))


# In[20]:


inv = {idx:word for word, idx in vocab.items()}


# In[21]:


(X_train, y_train), (Xt, yt) = data


# In[22]:


trainidx = [i for i, x in enumerate(X_train) if len(x) < 400]
trainidx, devidx = train_test_split(trainidx, train_size=0.8, random_state=1378)
X = [X_train[i] for i in trainidx]
y = [y_train[i] for i in trainidx]

Xd = [X_train[i] for i in devidx]
yd = [y_train[i] for i in devidx]

testidx = [i for i, x in enumerate(Xt) if len(x) < 400]
testidx, remaining_idx =  train_test_split(testidx, train_size=0.2, random_state=1378)

Xt = [Xt[i] for i in testidx]
yt = [yt[i] for i in testidx]


# In[23]:


def invert_and_join(X) :
    X = [[inv[x] for x in doc] for doc in X]
    X = [" ".join(x) for x in X]
    return X


# In[24]:


X = invert_and_join(X)
Xd = invert_and_join(Xd)
Xt = invert_and_join(Xt)


# In[25]:


texts = {'train' : X, 'test' : Xt, 'dev' : Xd}
labels = {'train' : y, 'test' : yt, 'dev' : yd}


# In[26]:


import pandas as pd
df_texts = []
df_labels = []
df_exp_splits = []

for key in ['train', 'test', 'dev'] :
    df_texts += texts[key]
    df_labels += labels[key]
    df_exp_splits += [key] * len(texts[key])
    
df = pd.DataFrame({'text' : df_texts, 'label' : df_labels, 'exp_split' : df_exp_splits})
df.to_csv('imdb_dataset.csv', index=False)


# In[27]:


get_ipython().run_line_magic('run', '"../preprocess_data_BC.py" --data_file imdb_dataset.csv --output_file ./vec_imdb.p --word_vectors_type fasttext.simple.300d --min_df 10')


# In[ ]:




