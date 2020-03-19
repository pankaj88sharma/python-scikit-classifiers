#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download()


# In[22]:


df = pd.read_csv('http://localhost:8983/solr/catalog/select?q=*:*&wt=csv&fl=title,department,category,product,brand&rows=9999999')


# In[23]:


df = df[pd.notnull(df['title'])]
df = df[pd.notnull(df['department'])]
df = df[pd.notnull(df['category'])]
df = df[pd.notnull(df['product'])]
df = df[pd.notnull(df['brand'])]


# In[24]:


df.shape


# In[25]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9_\s]+', '', text)
    text = text.strip(' ')
    return text
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(stemmed_tokens)


# In[26]:


df['TITLE'] = df['TITLE'].map(lambda x : clean_text(x))


# In[27]:


df['TITLE'] = df['TITLE'].map(lambda x: stem_sentences(x))


# In[28]:


df.head()


# In[29]:


col = ['TITLE', 'DEPARTMENT', 'CATEGORY', 'PRODUCT', 'BRAND']
df = df[col]


# In[30]:


df.to_csv('processed_catalog.csv', encoding='utf-8', index=False)


# In[ ]:




