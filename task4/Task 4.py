#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv('spamdetection.csv',encoding = "ISO-8859-1")


# In[3]:


print(data)


# In[4]:


data.shape


# In[5]:


data.isnull().sum(axis=0)


# In[6]:


data.info()


# In[7]:


data1=data.where((pd.notnull(data)),'')


# In[8]:


data1.isnull().sum(axis=0)


# In[44]:


data1['v3']=data1['v2']+data1['Unnamed: 2']+data1['Unnamed: 3']+data1['Unnamed: 4']


# In[18]:


print(data1)


# In[19]:


data1.drop(['v2','Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[20]:


data1.loc[data1['v1']=='spam','v1',]=0
data1.loc[data1['v1']=='ham','v1',]=1


# In[46]:


x1=data1['v3']
x2=data1['v1']


# In[49]:


print(x1)


# In[50]:


print(x2)


# In[51]:


x1_train,x1_test,x2_train,x2_test = train_test_split(x1,x2,test_size=0.2,random_state=3)


# In[52]:


print(x1.shape)
print(x1_train.shape)
print(x1_test.shape)


# In[60]:


feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
x1_train_feature=feature_extraction.fit_transform(x1_train)
x1_test_feature=feature_extraction.transform(x1_test)

x2_train= x2_train.astype('int')
x2_test=x2_test.astype('int')


# In[61]:


print(x1_train_feature)


# In[62]:


model=LogisticRegression()


# In[63]:


model.fit(x1_train_feature,x2_train)


# In[66]:


prediction_training=model.predict(x1_train_feature)
accuracy_train=accuracy_score(x2_train,prediction_training)


# In[67]:


print(accuracy_train)


# In[68]:


prediction_testing=model.predict(x1_test_feature)
accuracy_test=accuracy_score(x2_test,prediction_testing)


# In[69]:


print(accuracy_test)


# In[78]:


input_mail=["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]
input_feature=feature_extraction.transform(input_mail)
prediction=model.predict(input_feature)
print(prediction)
            
if (prediction[0]==1):
  print('ham mail')
else:
  print('spam mail')


# In[ ]:




