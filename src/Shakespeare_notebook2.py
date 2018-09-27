
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('C:\\Users\\p860n111\\Desktop\\data science\\Hw2\\Shakespeare_data.csv')


# In[3]:


# to see the total number of columns in the dataset.
df = df.dropna()
df.columns


# In[4]:


#to find the total number of unique play
print("Number of plays are: " + str(df['Play'].nunique()))


# In[5]:


# to calculate the total words in each PlayerLine cell entry
df['NoOfWordsInPlayerLine'] = df.PlayerLine.apply(lambda x: len(str(x).split(' ')))


# In[6]:


df


# In[6]:


df.dtypes


# In[7]:


df["Play"] = df["Play"].astype('category')
df.dtypes
df["Play_cat"] = df["Play"].cat.codes
df.head()


# In[8]:


df.dtypes


# In[9]:


df["PlayerLinenumber"] = df["PlayerLinenumber"].astype('category')
df.dtypes
df["PlayerLinenumber_cat"] = df["PlayerLinenumber"].cat.codes
df.head()
df["ActSceneLine"] = df["ActSceneLine"].astype('category')
df.dtypes
df["ActSceneLine_cat"] = df["ActSceneLine"].cat.codes
df.head()
df["Player"] = df["Player"].astype('category')
df.dtypes
df["Player_cat"] = df["Player"].cat.codes
df.head()


# In[10]:


df.dtypes


# In[11]:


df["PlayerLine"] = df["PlayerLine"].astype('category')
df.dtypes
df["PlayerLine_cat"] = df["PlayerLine"].cat.codes
df.head()


# In[12]:


df.dtypes


# In[13]:


newDF = df.filter(['Play_cat','PlayerLinenumber_cat','ActSceneLine_cat','Player_cat','PlayerLine_cat','NoOfWordsInPlayerLine' ], axis=1)


# In[16]:


newDF.dtypes


# In[14]:


X = newDF.ix[:,(0,1,2,4,5)].values
y = newDF.ix[:,3].values


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=25)


# In[17]:


from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)


# In[20]:


predictions = LogReg.predict(X_test)


# In[22]:


#Use score method to get accuracy of model
score = LogReg.score(X_test,  y_test)
print(score)


# In[23]:


newDF.dtypes


# In[33]:


X1 = newDF.ix[:,(0,1)].values
y1 = newDF.ix[:,3].values


# In[34]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = .3, random_state=25)


# In[35]:


LogReg.fit(X_train1, y_train1)


# In[37]:


predictions = LogReg.predict(X_test1)


# In[38]:


predictions


# In[39]:


#Use score method to get accuracy of model
score = LogReg.score(X_test1,  y_test1)
print(score)


# In[40]:




