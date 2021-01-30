#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# In[2]:


os.chdir(r"C:\Users\vashundhra\Downloads")
data=pd.read_csv('House_prediction.csv') 
data.shape


# In[3]:


data=data.replace('-',0)
data['floor']=data['floor'].astype('int64')
#le = LabelEncoder() 
#data['animal']= le.fit_transform(data['animal']).astype('int64') 
#data['furniture']= le.fit_transform(data['furniture']).astype('int64')
data.head(10)


# In[4]:


data.info()


# In[5]:


plt.figure(figsize=(10, 10))
#calculating correlation matrix and plotting hatmap through seaborn
corr = data.corr()
sns.heatmap(corr,  vmin=0, vmax=1, annot=True, linewidth=0.02, linecolor='black')


# In[6]:


plt.figure(figsize=(12, 6))
sns.boxplot( x=data['city'], y=data['rent amount (R$)'])


# In[7]:


plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
sns.barplot(x=data['bathroom'], y=data['rent amount (R$)'])
plt.subplot(1,2,2)
sns.boxplot(x=data['bathroom'])


# In[8]:


plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
sns.barplot(x=data['rooms'], y=data['rent amount (R$)'])
plt.subplot(1,2,2)
sns.boxplot(x=data['rooms'])


# In[9]:


plt.figure(figsize=(20, 6))
plt.subplot(1,2,1)
sns.barplot(x=data['floor'], y=data['rent amount (R$)'])
plt.subplot(1,2,2)
sns.boxplot(x=data['floor'])


# In[10]:


plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
sns.barplot(x=data['parking spaces'], y=data['rent amount (R$)'])
plt.subplot(1,2,2)
sns.boxplot(x=data['parking spaces'])


# In[11]:


plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
sns.boxplot(x=data['furniture'], y=data['rent amount (R$)'])
plt.subplot(1,2,2)
sns.boxplot(x=data['animal'], y=data['rent amount (R$)'])
no_furn = (data['furniture']=="not furnished").sum() #no.of non- furnished homes
furn = (data['furniture']== "furnished").sum()
no_ani = (data['animal']== "not acept").sum()
ani = (data['animal']== "acept").sum()


# In[12]:


print ( no_furn, furn, no_ani, ani)


# In[13]:


# remove outliers
new_data = pd.DataFrame()
new_data = data[data['rent amount (R$)']<=15000]
new_data.shape 


# In[14]:


plt.figure(figsize=(20, 5))
plt.subplot(1,2,1)
sns.boxplot(x=new_data['furniture'], y=new_data['rent amount (R$)'])
plt.subplot(1,2,2)
sns.boxplot(x=new_data['animal'], y=new_data['rent amount (R$)'])


# In[15]:


new_data.describe()


# In[42]:


#onehot = preprocessing.OneHotEncoder()
#feature_cat = ['rooms', 'bathroom', 'parking spaces','city', 'furniture']
#final_data = pd.DataFrame(new_data['rent amount (R$)'])
#for cat in feature_cat:
#    cat = pd.get_dummies(new_data[cat], prefix = cat)
#    final_data = final_data.join(cat)
#scaler = preprocessing.StandardScaler()
#final_data['fire insurance (R$)'] = scaler.fit_transform(pd.DataFrame(new_data['fire insurance (R$)']))
#final_data= final_data.drop(columns='rent amount (R$)')


# In[16]:


scaler = preprocessing.StandardScaler()
feature_cat = ['city', 'furniture']
final_data = pd.DataFrame(new_data['rent amount (R$)'])
for cat in feature_cat:
    cat = pd.get_dummies(new_data[cat])
    final_data = final_data.join(cat)
#final_data['fire insurance (R$)'] = scaler.fit_transform(pd.DataFrame(new_data['fire insurance (R$)']))
feature_cat = ['rooms', 'parking spaces','bathroom','fire insurance (R$)']
for cat in feature_cat:
    cat = new_data[cat]
    final_data = final_data.join(pd.DataFrame(cat))
final_data= final_data.drop(columns='rent amount (R$)') 
final_data = final_data.rename(columns={'fire insurance (R$)':'fire_insurance', 'not furnished': 'not_furnished', 'parking spaces': 'parking_spaces', 'Belo Horizonte': 'Belo_Horizonte' , 'Porto Alegre': 'Porto_Alegre', 'Rio de Janeiro': 'Rio_de_Janeiro', 'São Paulo': 'São_Paulo' })


# In[17]:


final_data.head()


# In[18]:


SEED = 42
X = final_data.values
y = new_data['rent amount (R$)'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)         


# In[19]:


regressor = [ RandomForestRegressor(), DecisionTreeRegressor(), SVR(), LinearRegression(), XGBRegressor(), GaussianNB() ] 
for reg in regressor:
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    mae= sum(abs(pred - y_test) ) / len(y_test)
    rmse= np.sqrt(sum( (pred - y_test)**2 ) / len(y_test))
    r2 = r2_score(y_test, pred)
    print ("Regressor : ", reg)
    print ("MAE : ", mae)
    print ("RMSE : ", rmse)
    print ("R2_score : ", r2)


# In[20]:


reg = XGBRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
r2 = r2_score(y_test, pred)
print ("R2_score : ", r2)


# In[21]:


plt.figure(figsize=(12, 8))
sns.distplot(y_test, hist=False, color='b', label ='Actual')
sns.distplot(pred, hist=False, color='r', label = 'Predicted')
plt.show()


# In[22]:


X_train.shape


# In[23]:


pred


# In[24]:


y_test


# In[ ]:


import pickle
pickle.dump(reg,open('model.pkl', 'wb'))
model = pickle.load (open('model.pkl', 'rb'))


# In[ ]:




