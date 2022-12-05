#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[2]:


cars=pd.read_csv('car data.csv')


# In[3]:


cars.head()


# In[5]:


cars.shape


# In[6]:


cars.info()


# In[7]:


cars.describe()


# In[8]:


cars.isnull().sum()


# In[9]:


fig=plt.figure(figsize=(8,4))
sns.distplot(cars['Selling_Price'])
plt.title('Sales data distribution')


# In[10]:


sns.set_theme(style="white")
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 9)),
                 columns=list(cars))
corr = d.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[11]:


print(cars['Seller_Type'].unique())
print(cars['Transmission'].unique())
print(cars['Owner'].unique())


# In[12]:


cars.columns


# In[13]:


final_dataset=cars[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[14]:


final_dataset['Current_Year']=2021


# In[15]:


final_dataset['no_year']=final_dataset['Current_Year']-final_dataset['Year']


# In[16]:


final_dataset.head()


# In[17]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[18]:


final_dataset.head()


# In[19]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[20]:


final_dataset.head()


# In[21]:


final_dataset=pd.get_dummies(final_dataset,drop_first=True)


# In[22]:


final_dataset.head()


# In[23]:


final_dataset.corr()


# In[24]:


corrmat=final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[25]:


y=final_dataset['Selling_Price']
y.head()


# In[26]:


X=final_dataset.drop(['Selling_Price'],axis=1)
X.head()


# In[27]:


model=ExtraTreesRegressor()
model.fit(X,y)


# In[28]:


print(model.feature_importances_)


# In[29]:


feat_importances=pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[30]:


scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# In[32]:


random_forest_regressor=RandomForestRegressor()
random_forest_regressor.fit(X_train,y_train)


# In[33]:


train_acc=random_forest_regressor.score(X_train,y_train)
test_acc=random_forest_regressor.score(X_test,y_test)
print('Training Accuracy: ',round(train_acc*100, 2),'%')
print('Testing Accuracy: ',round(test_acc*100, 2),'%')


# In[34]:


predictions=random_forest_regressor.predict(X_test)
predictions


# In[35]:


sns.distplot(y_test-predictions)


# In[ ]:


plt.scatter(y_test,predictions)

