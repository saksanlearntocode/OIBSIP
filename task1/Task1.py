#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from warnings import filterwarnings
filterwarnings(action='ignore')


# In[45]:


data=pd.read_csv("Iris.csv")


# In[46]:


data


# In[47]:


print(iris.describe())


# In[48]:


print(iris.isna().sum())


# In[49]:


data.head()


# In[50]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[51]:


iris.hist()
plt.show()


# In[52]:


sns.pairplot(iris,hue='Species');


# In[53]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[54]:


train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[55]:


train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
train_y = train.Species

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
test_y = test.Species


# In[56]:


train_X.head()


# In[57]:


test_y.head()


# In[58]:


model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print(metrics.accuracy_score(prediction,test_y))


# In[59]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# In[60]:


from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# In[61]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[62]:


from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))


# In[63]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[69]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(train_X,train_y)
y_pred=knn.predict(test_X)
print(metrics.accuracy_score(test_y,y_pred2))


# In[74]:


k_range=range(1,100)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X,train_y)
    y_pred=knn.predict(test_X)
    scores.append(metrics.accuracy_score(test_y,pred_y))


# In[79]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(train_X,train_y)
knn.predict([[3,5,4,2],[5.1, 3.5, 1.4, 0.2]])


# In[ ]:




