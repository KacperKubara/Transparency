#!/usr/bin/env python
# coding: utf-8

# ## Downloading the Dataset
# 
# Download and extract the 'yelp_review_full_csv.tar.gz' file from https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
# 
# Make sure that 'train.csv' and 'test.csv' files are present in this directory

# In[11]:


import os

assert os.path.exists('train.csv') and os.path.exists('test.csv')


# In[13]:


get_ipython().system('export PYTHONIOENCODING=utf8')


# In[14]:


import nltk
import pandas as pd
import csv
import sys
import spacy
import re
import random
import codecs
from importlib import reload

random.seed(1357)
def read_input_file(input_file):
    lines = csv.reader(codecs.open(input_file, "r", encoding="utf-8"))
    lines = list(lines)
    random.shuffle(lines)
    new_labels = []
    new_lines = []
    for label, line in lines:
        if int(label) < 3:
            new_labels.append("0")
            new_lines.append(line)
        elif int(label) > 3:
            new_labels.append("1")
            new_lines.append(line)
            
    print (new_labels[:2], new_lines[:2])
    print(len(new_labels), len(new_lines))
    return new_labels, new_lines
                


# In[15]:


labels_train, content_train = read_input_file("train.csv")
assert(len(labels_train) == len(content_train))
print (len(labels_train))

labels_dev, content_dev = labels_train[:7000], content_train[:7000]
keys_dev = ["dev"]* len(labels_dev)

labels_train, content_train = labels_train[7000:], content_train[7000:]
keys_train = ["train"]*len(labels_train)


# In[16]:


labels_test, content_test = read_input_file("test.csv")
keys_test = ["test"]*len(labels_test)
assert(len(labels_test) == len(content_test))
print (len(labels_test))


# In[17]:


content_train[:10]


# In[ ]:


nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

def tokenize(text) :
    #text = " ".join(text)
    text = text.replace("-LRB-", '')
    text = text.replace("-RRB-", " ")
    text = text.strip()
    tokens = " ".join([t.text.lower() for t in nlp(text)])
    return tokens

labels = [int(i) for i in labels_train]
content = [tokenize(i) for i in content_train]

assert(len(labels) == len(content))
labels[:3]
content[:3]


# In[ ]:


content[0]
labels = labels_train + labels_dev + labels_test
content = content_train + content_dev + content_test
keys = keys_train + keys_dev + keys_test


# In[ ]:


df = pd.DataFrame({'text' : content, 'label' : labels, 'exp_split' : keys})
df.to_csv('yelp_dataset.csv', index=False)

df = {'paragraph' : df_paragraphs, 'question' : df_questions, 'answer' : df_answers, 'exp_split' : df_exp_splits}
df = pd.DataFrame(df)


# In[2]:


get_ipython().run_line_magic('run', '"../preprocess_data_BC.py" --data_file yelp_dataset.csv --output_file ./vec_yelp.p --word_vectors_type fasttext.simple.300d --min_df 20')


# In[ ]:




