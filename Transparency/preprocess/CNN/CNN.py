#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://cs.stanford.edu/~danqi/data/cnn.tar.gz')
get_ipython().system('tar -xvzf cnn.tar.gz')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


keys = ['train', 'dev', 'test']


# In[4]:


p, q, a = {}, {}, {}
for k in keys :
    file = open('cnn/' + k + '.txt').read().strip().split('\n\n')
    file = [x.split('\n') for x in file]
    p[k] = [x[2] for x in file]
    q[k] = [x[0] for x in file]
    a[k] = [x[1] for x in file]


# In[5]:


entities = {}
for k in p :
    entities[k] = []
    for x in p[k] :
        entities[k] += [y for y in x.split() if y.startswith('@entity')]
    
    entities[k] = set(entities[k])
    
f = open('entity_list.txt', 'w')
f.write('\n'.join(list(entities['train'])))
f.close()


# In[6]:


def generate_possible_answers(p) :
    possible_answers = []
    for w in p.split() :
        if w.startswith('@entity') :
            possible_answers.append(w)
    
    return ";".join(list(set(possible_answers)))


# In[7]:


import pandas as pd
df_paragraphs = []
df_questions = []
df_answers = []
df_possible_answers = []
df_exp_splits = []

for k in keys :
    df_paragraphs += p[k]
    df_questions += q[k]
    df_answers += a[k]
    df_possible_answers += [generate_possible_answers(x) for x in p[k]]
    df_exp_splits += [k] * len(p[k])
    
df = {'paragraph' : df_paragraphs, 'question' : df_questions, 'answer' : df_answers, 
      'exp_split' : df_exp_splits, 'possible_answers' : df_possible_answers}
df = pd.DataFrame(df)


# In[8]:


df.to_csv('cnn_dataset.csv', index=False)


# In[9]:


get_ipython().run_line_magic('run', '"../preprocess_data_QA.py" --data_file cnn_dataset.csv --output_file ./vec_cnn.p --all_answers_file entity_list.txt --word_vectors_type fasttext.simple.300d --min_df 8 --add_answers_to_vocab')


# In[ ]:




