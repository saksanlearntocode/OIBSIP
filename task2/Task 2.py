#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[7]:


data = pd.read_csv("unemployment_Rate.csv")
print(data.head())


# In[4]:


data.isnull().sum(axis=0)


# In[21]:


data.describe()


# In[9]:


plt.style.use("seaborn-whitegrid")#  Correlation between dataset
plt.figure(figsize=(12,10))
sns.heatmap(data.corr())
plt.show()


# In[12]:


# Looking at the estimated no of employes according to the diffrent regions of india
data.columns=["States","Date","Frequency","Estimated Unemployment Rate","Estimated Employed","Estimated Labour Participation Rate","Region","Longitude","Latitude"]
plt.title("India Unemployment")
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.show()


# In[14]:


#  See the unemployment rate in diffrent regions of India
plt.figure(figsize=(12,10))
plt.title("India Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[16]:


unemployment= data[["States","Region","Estimated Unemployment Rate"]]
figure = px.sunburst(unemployment, path=["Region","States"],
                     values="Estimated Unemployment Rate",
                     width=700,height=700,color_continuous_scale="RdY1Gn",
                     title="Unemployment Rate in India")
figure.show()


# In[17]:




