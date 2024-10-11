#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Exploration & Cleaning:

# In[2]:


df = pd.read_csv("WALMART_SALES_DATA.csv")
data_shallow = df
data_shallow 


# In[3]:


data_shallow.describe()


# In[4]:


data_shallow.info()


# In[5]:


data_shallow.columns


# In[6]:


data_shallow.shape


# # Handling Null Values

# In[7]:


data_shallow.isnull().sum()


# # Outliers Analysis

# ## (1) Which store has maximum sales?

# In[8]:


total_sales = df.groupby('Store')['Weekly_Sales'].sum().round().sort_values(ascending=False)


# In[9]:


# Maximum Sales

pd.DataFrame(total_sales).head(1)


# In[10]:


# Minimum Sales

pd.DataFrame(total_sales).tail(1)


# ## (2) Which store has maximum standard deviation i.e., the sales vary a lot. Also, find out the coefficient of mean to standard deviation

# In[11]:


df_std = df.groupby('Store')['Weekly_Sales'].std().round(3).sort_values(ascending=False)


# In[12]:


pd.DataFrame(df_std).head()


# ### Store - 14 has a maximum standard deviation = $317,569.949

# In[13]:


store14 = df[df.Store == 14].Weekly_Sales


# In[14]:


mean_to_stddev = store14.std()/store14.mean()*100

mean_to_stddev.round(2)


# ### Mean to Standard Deviation = 15.71%

# ## (3) Which store/s has good quarterly growth rate in Q3’2012 ?

# In[15]:


q2_sales = df[(df['Date'] >= '2012-04-01') & (df['Date'] <= '2012-06-30')].groupby('Store')['Weekly_Sales'].sum().round()
q3_sales = df[(df['Date'] >= '2012-07-01') & (df['Date'] <= '2012-09-30')].groupby('Store')['Weekly_Sales'].sum().round()


# In[16]:


pd.DataFrame({'Q2 Sales':q2_sales,
              'Q3 Sales':q3_sales,
              'Difference':(q3_sales-q2_sales),
              'Growth Rate':(q3_sales-q2_sales)/q3_sales*100}).sort_values(by=['Growth Rate'], ascending=False).head()


# ## (4) Some holidays have a negative impact on sales. Find out holidays which have higher¶ sales than the mean sales in non-holiday season for all stores together.

# In[17]:


# Holiday Events

Super_Bowl = ['12-2-2010', '11-2-2011', '10-2-2012']
Labour_Day =  ['10-9-2010', '9-9-2011', '7-9-2012']
Thanksgiving =  ['26-11-2010', '25-11-2011', '23-11-2012']
Christmas = ['31-12-2010', '30-12-2011', '28-12-2012']


# In[18]:


# Calculating holiday events sales

Super_Bowl_sales = df.loc[df.Date.isin(Super_Bowl)]['Weekly_Sales'].mean()
Labour_Day_sales = df.loc[df.Date.isin(Labour_Day)]['Weekly_Sales'].mean()
Thanksgiving_sales = df.loc[df.Date.isin(Thanksgiving)]['Weekly_Sales'].mean()
Christmas_sales = df.loc[df.Date.isin(Christmas)]['Weekly_Sales'].mean()


# In[19]:


Super_Bowl_sales, Labour_Day_sales, Thanksgiving_sales, Christmas_sales


# In[20]:


non_holiday_sales = df[(df['Holiday_Flag'] == 0)]['Weekly_Sales'].mean().round(2)
non_holiday_sales


# In[21]:


result = pd.DataFrame([{'Super Bowl Sales':Super_Bowl_sales,
              'Labour Day Sales':Labour_Day_sales,
              'Thanksgiving Sales':Thanksgiving_sales,
              'Christmas Sales':Christmas_sales,
              'Non Holiday Sales':non_holiday_sales}]).T

result


# ## (5) Provide a monthly and semester view of sales in units and give insights.

# In[22]:


df['Day'] = pd.DatetimeIndex(df['Date']).day
df['Month'] = pd.DatetimeIndex(df['Date']).month
df['Year'] = pd.DatetimeIndex(df['Date']).year


# In[23]:


df.head()


# In[24]:


# Sales for the Year - 2010

plt.figure(figsize=(14,7), dpi=80)
graph1 = sns.barplot(data=df, x=df[df.Year==2010]['Month'], y=df[df.Year==2010]['Weekly_Sales'])
graph1.set(title='Monthwise Sales for 2010')

for p in graph1.patches:
    graph1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 35), textcoords = 'offset points')


# In[25]:


# Sales for the Year - 2011

plt.figure(figsize=(14,7), dpi=80)
graph1 = sns.barplot(data=df, x=df[df.Year==2011]['Month'], y=df[df.Year==2011]['Weekly_Sales'])
graph1.set(title='Monthwise Sales for 2011')

for p in graph1.patches:
    graph1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 35), textcoords = 'offset points')


# In[26]:


# Sales for the Year - 2012

plt.figure(figsize=(14,7), dpi=80)
graph1 = sns.barplot(data=df, x=df[df.Year==2012]['Month'], y=df[df.Year==2012]['Weekly_Sales'])
graph1.set(title='Monthwise Sales for 2012')

for p in graph1.patches:
    graph1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 35), textcoords = 'offset points')


# In[27]:


# Monthwise Sales

plt.figure(figsize=(14,7), dpi=80)
plt.bar(df['Month'], df['Weekly_Sales'])
plt.xlabel('Months')
plt.ylabel('Weekly_Sales')
plt.title('Monthwise Sales')


# In[28]:


# Yearly Sales

plt.figure(figsize=(10,7), dpi=80)
df.groupby('Year')[['Weekly_Sales']].sum().plot(kind='bar', legend=False)
plt.title('Yearly Sales')


# ### (1) Year 2010 has the highest sales and 2012 has the lowest sales.

# ### (2) December month has the highest weekly sales.

# ### (3) Year 2011 has the highest weekly sales.

# # Handling Outliers

# In[29]:


fig, axis = plt.subplots(4, figsize=(12,16), dpi=80)
x = df[['Temperature','Fuel_Price','CPI','Unemployment']]

for i, column in enumerate(x):
    sns.boxplot(df[column], ax=axis[i])

import warnings
warnings.filterwarnings('ignore')


# In[30]:


df.info()


# # Splitting into x and y

# In[31]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import category_encoders as ce
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


X = df[['Store','Fuel_Price','CPI','Unemployment','Day','Month','Year']]
Y = df['Weekly_Sales']


# In[33]:


X.shape, Y.shape


# In[34]:


X


# # Encoding Categorical Features

# In[35]:


encoder = ce.LeaveOneOutEncoder()
x=encoder.fit_transform(X,Y)


# In[36]:


Y


# In[37]:


X.info()


# In[38]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


# # Model Building

# In[39]:


model_rfr=RandomForestRegressor(n_estimators=7)
model_dt=DecisionTreeRegressor()
model_lr=LinearRegression()


# In[40]:


models=[model_rfr,model_dt,model_lr]


# In[41]:


#Fitting model:
for model in models:
    print(f"fitting model: {model}")
    model.fit(X_train,Y_train)


# In[42]:


for model in models:
    print(f"score of {model} for training data: {model.score(X_train,Y_train)}")


# In[43]:


for model in models:
    print(f"score of {model} for testing data: {model.score(X_test,Y_test)}")


# In[44]:


#Feature importance:
fs= model_rfr.feature_importances_
feature_names= X.columns


# In[45]:


feature_importances= pd.DataFrame(fs,feature_names).sort_values(by=0,ascending=False)
plt.figure(figsize=(12,9))
plt.title("Feature Importances")
plt.bar(x=feature_importances.index,height=feature_importances[0])
plt.xticks(rotation=90)
plt.show()


# In[46]:


feature_importances


# In[47]:


#Train and Test using 
def regression_results(Y_true,Y_pred):
    #Regression metrics
    #explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(Y_true,Y_pred)
    mse=metrics.mean_squared_log_error(Y_true,Y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(Y_true,Y_pred)
    r2=metrics.r2_score(Y_true,Y_pred)
    
    #print('explained_variance: ',round(explained_variance,4))
    #print('mean_squared_log_error: ',round(mean_squared_log_error,4))
    print('r2:',round(r2,4))
    print('MAE:',round(mean_absolute_error,4))
    print('MSE:',round(mse,4))
    print('RMSE:',round(np.sqrt(mse),4))
    #print('Median absolute error: ',round(medium_absolute_error,4))


# In[48]:


for model in models[:]:
    Y_predicted = model.predict(X_test)
    
    print(f"Report:{model}")
    print(f"{regression_results(Y_test,Y_predicted)}\n")


# ### KNearest Neighbors

# In[49]:


from sklearn.neighbors import KNeighborsRegressor
print('KNeighborsRegressor:')
print()
knn = KNeighborsRegressor()        
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)

print('r2:',knn.score(X_test, Y_test)*100)
print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

import warnings
warnings.filterwarnings('ignore')


# # Cross Validation

# In[50]:


from sklearn.model_selection import cross_val_score


# In[51]:


# Linear Regression

lr_scores = cross_val_score(model_lr, X_train,Y_train, cv=10, scoring='r2')
print(lr_scores)
print("Mean Score:", lr_scores.mean()*100,'%')


# In[52]:


# Random Forest Regression

rfr_scores = cross_val_score(model_rfr, X_train,Y_train, cv=10, scoring='r2')
print(rfr_scores)
print("Mean Score:", rfr_scores.mean()*100,'%')


# In[53]:


# Decision Tree Regression

dtr_scores = cross_val_score(model_dt, X_train,Y_train, cv=10, scoring='r2')
print(dtr_scores)
print("Mean Score:", dtr_scores.mean()*100,'%')


# In[54]:


# KNearest Neighbor

knn_scores = cross_val_score(knn, X_train,Y_train, cv=10, scoring='r2')
print(knn_scores)
print("Mean Score:", knn_scores.mean()*100,'%')


# # Conclusion

# ### Here, we have used 4 different algorithms to know which model to use to predict the weekly sales. Linear Regression is not an appropriate model to use as accuracy is very low. However, Random Forest Regression gives accuracy of almost 95% . so, it is the best model to forecast weekly sales.¶
