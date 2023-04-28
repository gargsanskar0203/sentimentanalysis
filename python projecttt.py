#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import nltk


# In[20]:


df=pd.read_csv('train.csv', encoding='unicode_escape')
df


# In[21]:


df=df.drop(['textID', 'text', 'Time of Tweet', 'Age of User', 'Country'], axis=1)
df=df.drop(['Population -2020', 'Land Area (Km²)', 'Density (P/Km²)'], axis=1)
df


# In[22]:


df['sentiments']=np.where(df['sentiment']=='negative',-1, 1)
df


# In[23]:


df['selected_text']=df['selected_text'].str.lower()
df


# In[24]:


import nltk
nltk.download('stopwords')


# In[25]:


stop_words = set(nltk.corpus.stopwords.words('english'))
df['selected_text'].astype(str)


# In[26]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


# In[27]:


vec=CountVectorizer()
df=df.dropna()
x=vec.fit_transform(df['selected_text'])


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, df['sentiments'], test_size=0.2, random_state=42)


# In[29]:


lr=LogisticRegression()
lr.fit(x_train, y_train)


# In[30]:


y_pred = lr.predict(x_test)


# In[31]:


accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)


# In[32]:


print(f"Accuracy: {round(accuracy*100,2)}%")
print(f"Precision: {round(precision*100,2)}%")
print(f"Recall: {round(recall*100,2)}%")
print(f"F1 Score: {round(f1*100,2)}%")


# In[ ]:





# In[ ]:





# In[ ]:




