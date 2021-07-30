#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
my_first_dataframe = pd.read_csv('kddcup99_csv.csv')
col_names = my_first_dataframe.columns
#my_first_dataframe.head()
#my_first_dataframe.info(memory_usage='deep')
my_first_dataframe=my_first_dataframe.iloc[::30, :]
#my_first_dataframe.info(memory_usage='deep')
my_first_dataframe.head()
my_first_dataframe.dtypes


# In[3]:


my_first_dataframe.drop(columns=['protocol_type', 'service','flag'],inplace=True)
my_first_dataframe.dtypes


# In[11]:


my_first_dataframe['label']


# In[5]:


X = my_first_dataframe.drop(columns=['label']) # Features
y = my_first_dataframe.label # Target variable


# In[26]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.999,random_state=0)


# In[27]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(C=2,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=0.5,solver='liblinear', max_iter=1000, tol=0.0001, verbose=0,warm_start=False, multi_class='ovr',n_jobs=1)
#C=2,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=0.5,solver='liblinear', max_iter=100, tol=0.0001, verbose=0,warm_start=False, multi_class='ovr',n_jobs=1


# In[28]:


# fit the model with data
logreg.fit(X_train,y_train)


# In[29]:


#
y_pred=logreg.predict(X_test)
#y_pred.shape


# In[30]:


score = logreg.score(X_test,y_test)
print(score)


# In[39]:


logreg.coef_


# In[48]:


logreg.coef_[0][20]
X_train.columns[20]
X_train['srv_count'].unique()


# In[43]:


#y_train
X_train
#X_test
#y_test
#y_train.groupby(['label']).size() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




