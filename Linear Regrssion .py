#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
from sklearn import linear_model
df=pd.read_csv("new.csv")
df


# In[53]:


df.head(10)


# In[54]:


df.drop(['charges','region'] ,axis='columns',inplace=True)
df.head(10)


# In[51]:


inputs = df.drop('sex',axis='columns')
target = df.sex


# In[55]:


dummies = pd.get_dummies(inputs.smoker)
dummies.head(10)


# In[56]:


inputs=pd.concat([inputs,dummies],axis='columns')
inputs.head(10)


# In[62]:


inputs.drop(['smoker','yes'],axis='columns',inplace=True)
inputs.head(10)


# In[71]:


reg = linear_model.LinearRegression()
reg.fit(inputs.drop('age',axis='columns'),inputs.age)


# In[72]:


reg.coef_


# In[73]:


reg.predict([[39, 34000, 1]])


# In[68]:


reg.fit(inputs.drop('bmi',axis='columns'),inputs.bmi)


# In[74]:


reg.coef_


# In[75]:


reg.predict([[60,34000,0]])


# In[1]:





# In[ ]:




