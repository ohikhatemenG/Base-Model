#!/usr/bin/env python
# coding: utf-8

#                              THE CLIMATE CHANGE DATASET
# The datasets was provided by the Food and Agriculture Organization of the United Nations 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
url = "https://github.com/HamoyeHQ/HDSC-Introduction-to-Python-for-machine-learning/files/7768140/FoodBalanceSheets_E_Africa_NOFLAG.csv"

dat = pd.read_csv(url, encoding='latin-1')


# In[2]:


dat.head()


# In[3]:


# check the datasets numbers of rows and colums
dat.shape


# In[4]:


# check the types of data that made up the dataset
dat.info()


# In[6]:


# check for missing data
dat.isnull().sum()


# We observed that there was filling data in year 2014, 2015, 2016, 2017 and 2018 

# In[7]:


# let fill in the missing data in the datasets
dat.Y2014 = dat.Y2014.fillna(method='ffill')
dat.Y2015 = dat.Y2015.fillna(method='ffill')
dat.Y2016 = dat.Y2016.fillna(method='ffill')
dat.Y2017 = dat.Y2017.fillna(method='ffill')
dat.Y2018 = dat.Y2018.fillna(method='ffill')


# In[8]:


# Let check to confirm the results above
dat.isna().sum()


# In[9]:


# Let check the summary statistics of the datasets
dat.describe()


# In[10]:


# Let check the correlation of the dataset
dat.corr()


# In[11]:


# Let group the areas of the datasets
dat.groupby('Area')['Area'].count()


# In[13]:


# Let check the total production of each country for year 2017 
dat.groupby(['Area'], as_index=False).Y2017.count()


# Kenya has the hightest sum of Agricultural production of 1560 and while Ethiopia has the smallest with 39 in the year 2017

# In[15]:


#Let group the unit of the datasets base on measurement used
dat.groupby('Unit')['Unit'].count()


# In[17]:


# Let group the Element of production for the year 2014
dat.groupby(['Element']).Y2014.count()


# In[18]:


# Let group the Element of production for the year 2015
dat.groupby(['Element']).Y2015.count()


# In[19]:


# Let group the Element of production for the year 2016
dat.groupby(['Element']).Y2016.count()


# In[20]:


# Let group the Element of production for the year 2017
dat.groupby(['Element']).Y2017.count()


# In[21]:


# Let group the Element of production for the year 2018
dat.groupby(['Element']).Y2018.count()


# In[23]:


# Let group the items in the datasets
dat.groupby('Item')['Item'].count()


# VISUALIZATION OF THE DATASETS

# In[25]:


# Let plot the area of the dataset in bar chart
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
dat_area = pd.DataFrame({'Area':['Benin','Sudan','Togo','Nigeria','South Africa','Comoros'],
                        'count':[1285, 1316, 1294, 1474, 1399, 45]})
sns.barplot(data=dat_area, x='Area', y='count')
plt.xlabel('Climate Change Area')


# In[26]:


# Let plot the area of production for the year 2018

plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)

dat_ele = pd.DataFrame({'Element':['Domestic supply quantity','Export Quantity','Fat supply quantity',
                                   'Feed','Food','Food supply (kcal/capita/day)','Food supply quantity (kg/capita/yr)',
                                   'Import Quantity','Losses','Other uses','Processing','Production ','Protein supply quantity',
                                   'Residuals','Seed','Stock Variation','Total Population','Tourist consumption'], 
                        'count':['5295','4403','5023','1319','4941','5014','4905','5139','2009','1732','2010','3881','5023',
                                '4655','762','4232','45','555']
                       })
sns.barplot(data=dat_ele, x='Element', y='count')
plt.xlabel('Climate Change Element')


# In[31]:


dat.Y2014.plot(figsize=(10,5), title='Y2014')


# In[ ]:




