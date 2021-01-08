#!/usr/bin/env python
# coding: utf-8

# ### Downloading the Quora Question Paraphrase (QQP) Dataset
#  
# Download and extract the QQP dataset from https://gluebenchmark.com/task 
# 
# You should have the following files QQP/train.tsv, QQP/test.tsv, QQP/dev.tsv

# In[10]:


import os
assert os.path.exists('QQP/train.tsv') and os.path.exists('QQP/test.tsv') and os.path.exists('QQP/dev.tsv')


# In[1]:


import nltk
import pandas as pd
import csv
import sys
import spacy
import re
import random
from random import shuffle
import codecs
import numpy as np
from tasks import QQPTask
random.seed(0)


# In[2]:


NAME2INFO = {'qqp': (QQPTask, 'QQP')}


# In[13]:


def preprocess(name,max_seq_len):
    
    task = NAME2INFO[name][0](NAME2INFO[name][1], max_seq_len, name)
    train_data = task.train_data_text
    
    train = list(zip(train_data[0],train_data[1],train_data[2]))
    total_len = len(train)
    val_len = int(total_len*0.1)

    val = list(zip(*train[:val_len]))
    train = list(zip(*train[val_len:]))
    test = task.val_data_text

    print ("Train datapoints",len(train[0]))
    print ("Test datapoints",len(test[0]))
    print ("Val datapoints",len(val[0]))

    df_paragraphs = list(train[1]) + list(test[1]) + list(val[1])
    df_questions = list(train[0]) + list(test[0]) + list(val[0])
    df_answers = list(train[2]) + list(test[2]) + list(val[2])
    df_exp_splits = ['train']*len(train[0]) + ['test']*len(test[0]) + ['dev']*len(val[0])
        
    entity_list = [str(i) for i in np.unique(np.array(df_answers))]
    f = open('{}/entity_list.txt'.format(NAME2INFO[name][1]), 'w')
    f.write("\n".join(entity_list))
    f.close()
    df = {'paragraph' : df_paragraphs, 'question' : df_questions, 'answer' : df_answers, 'exp_split' : df_exp_splits}
    df = pd.DataFrame(df)
    df = df.dropna()
    df.to_csv('{}/{}_dataset.csv'.format(NAME2INFO[name][1],name), index=False)


# In[4]:


name="qqp"
max_seq_len=40
preprocess(name,max_seq_len)


# In[9]:


data_file = '{}/{}_dataset.csv'.format(NAME2INFO[name][1],name)
output_file = 'vec_{}.p'.format(name)
answers_file = '{}/entity_list.txt'.format(NAME2INFO[name][1])

# %run "../preprocess_data_QA.py" --data_file $data_file --output_file $output_file --all_answers_file $answers_file --word_vectors_type glove.840B.300d --min_df 10
get_ipython().run_line_magic('run', '"../preprocess_data_QA.py" --data_file $data_file --output_file $output_file --all_answers_file $answers_file --word_vectors_type glove.840B.300d --min_df 5')


# In[ ]:





# In[ ]:





# In[ ]:




