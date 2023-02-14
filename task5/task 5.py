#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data=pd.read_csv("Advertising.csv")


# In[ ]:


data


# In[ ]:


del data['Unnamed: 0']
data


# In[ ]:


X = data.drop(['Sales'], axis=1).values
y = data['Sales'].values.reshape(-1,1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[ ]:


y_pred=regressor.predict(X_test)


# In[ ]:


print("Coefficient = ",regressor.coef_)
print("Intercept = ",regressor.intercept_)
from sklearn.metrics import r2_score
print("R2 Score = %.2f"%r2_score(y_test, y_pred))
print("Sales = %.2f"%regressor.predict([[45,50,75]]))


# In[ ]:


print("Enter the ammount you will invest on:")
tv = float(input("TV : "))
radio = float(input("Radio : "))
newspaper = float(input("Newspaper : "))


# In[ ]:


output = regressor.predict([[tv,radio,newspaper]])
print("Sales = %.2f"%output)


# In[ ]:




