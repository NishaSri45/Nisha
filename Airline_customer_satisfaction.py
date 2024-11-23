#!/usr/bin/env python
# coding: utf-8

# # About Dataset

# The dataset provides insights into customer satisfaction levels within an undisclosed airline company. While the specific airline name is withheld, the dataset is rich in information, containing 22 columns and 129,880 rows. It aims to predict whether future customers will be satisfied based on various parameters included in the dataset.
# 
# The columns likely cover a range of factors that influence customer satisfaction, such as flight punctuality, service quality, and so. By analyzing this dataset, airlines can gain valuable insights into the factors that contribute to customer satisfaction and tailor their services accordingly to enhance the overall customer experience.

# - **Satisfaction:** Indicates the satisfaction level of the customer.
# - **Customer Type:** Type of customer: 'Loyal Customer' or 'Disloyal Customer’.
# - **Age:** Age of the customer.
# - **Type of Travel:** Purpose of the travel: 'Business travel' or 'Personal Travel’.
# - **Class:**	Class of travel: 'Business', 'Eco', or 'Eco Plus’.
# - **Flight Distance:** The distance of the flight in kilometres
# - **Seat comfort:** Rating of seat comfort provided during the flight (1 to 5).
# - **Departure/Arrival time convenient** Rating of the convenience of departure/arrival time (1 to 5).
# - **Food and drink:** Rating of food and drink quality provided during the flight (1 to 5).
# - **Gate location:**	Rating of gate location convenience (1 to 5).
# - **Inflight wifi service:**	Rating of inflight wifi service satisfaction (1 to 5).
# - **Inflight entertainment:** Rating of inflight entertainment satisfaction (1 to 5).
# - **Online support:** Rating of online customer support satisfaction (1 to 5).
# - **Ease of Online booking:** Rating of ease of online booking satisfaction (1 to 5).
# - **On-board service:** Rating of on-board service satisfaction (1 to 5).
# - **Leg room service:** Rating of leg room service satisfaction (1 to 5).
# - **Baggage handling:** Rating of baggage handling satisfaction (1 to 5).
# - **Checkin service:** Rating of check-in service satisfaction (1 to 5).
# - **Cleanliness:** Rating of cleanliness satisfaction (1 to 5).
# - **Online boarding:** Rating of online boarding satisfaction (1 to 5).
# - **Departure Delay in Minutes:** Total departure delay in minutes.
# - **Arrival Delay in Minutes:** Total arrival delay in minutes.

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay, classification_report
warnings.filterwarnings('ignore')


# # Functions for plot customizations 

# In[2]:


def set_size_style(width, height, style=None):
    plt.figure(figsize=(width, height))
    if style != None:
        sns.set_style(style)

def customize_plot(plot, title:str, xlabel:str,  ylabel:str, title_font:int, label_font:int):
    plot.set_title(title, fontsize = title_font, weight='bold')
    plot.set_xlabel(xlabel, fontsize = label_font, weight='bold')
    plot.set_ylabel(ylabel, fontsize = label_font, weight='bold')


# # Data Exploration & Cleaning

# In[3]:


customer_df = pd.read_csv('Airline_customer_satisfaction.csv')
customer_df.head()


# In[4]:


customer_df.shape


# - There are 129880 row and 22 columns in the dataset

# In[5]:


customer_df.info()


# - The majority of the columns in the dataset consist of numeric values, primarily representing ratings.

# In[6]:


customer_df.describe()


# In[7]:


customer_df.describe(include = 'object')


# In[8]:


for col in customer_df.describe(include='object').columns:
    print('Column Name: ',col)
    print(customer_df[col].unique())
    print('-'*50)


# ## Handling Null Values

# In[9]:


customer_df.isna().sum()


# In[10]:


customer_df['Arrival Delay in Minutes'].fillna(customer_df['Arrival Delay in Minutes'].mean(), inplace=True) 


# ## Handling Outliers

# In[11]:


for col in customer_df.describe().columns:
    set_size_style(16,2,'ticks')
    sns.boxplot(data=customer_df, x=col)
    plt.show()


# In[12]:


customer_df = customer_df.drop(customer_df[customer_df['Departure Delay in Minutes'] > 500 ].index)
customer_df = customer_df.drop(customer_df[customer_df['Arrival Delay in Minutes'] > 500 ].index)
customer_df = customer_df.drop(customer_df[customer_df['Flight Distance'] > 5500 ].index)
customer_df.reset_index(drop=True, inplace=True)
customer_df.shape


# # Exploratory Data Analysis

# In[13]:


customer_df.columns


# In[14]:


set_size_style(10,5)
ax = sns.histplot(customer_df['Age'],bins=25,color= sns.color_palette('Spectral')[0],kde=True)
customize_plot(ax,'Age Distribution','Age','Frequency',13,10)


# - The majority of individuals fall within the age range of 20 to 60 years, with a notable concentration around the age of 40.

# In[15]:


plt.title("Satisfied vs Dissatisfied", fontsize = 12, weight='bold')
plt.pie(customer_df['satisfaction'].value_counts(),labels=customer_df['satisfaction'].value_counts().index,radius=1, autopct='%.2f%%',textprops={'fontsize': 10, 'fontweight': 'bold'}, colors = sns.color_palette('Spectral'))
plt.show()


# - The number of satisfied customers exceeds the number of dissatisfied customers, indicating a prevailing trend towards positive experiences with the service or product.

# In[16]:


set_size_style(12,5)
age_groups = customer_df.groupby('Age')['satisfaction'].value_counts(normalize=True).unstack()
satisfied_percentage = age_groups['satisfied'] * 100
ax =sns.lineplot(x=satisfied_percentage.index, y=satisfied_percentage.values, marker='o', color= sns.color_palette('Spectral')[0])
customize_plot(ax, 'Satisfied Percentage across Age', 'Age', 'Satisfied Percentage',13,10)
plt.grid(True)
plt.show()


# - Individuals in their 40s and 50s exhibit satisfaction with airline services.
# - Conversely, older individuals above the age of 70 express significantly higher levels of dissatisfaction with the services provided

# In[17]:


set_size_style(12,7)
class_ratings = customer_df.groupby('Class').agg({'Cleanliness':'mean',
                                                       'Checkin service' : 'mean',
                                                       'Seat comfort':'mean',
                                                       'Inflight wifi service':'mean', 
                                                       'Leg room service':'mean'}).reset_index()
class_ratings_melted = pd.melt(class_ratings, id_vars='Class', var_name='Category', value_name='Mean Rating')
ax = sns.barplot(x='Class', y='Mean Rating', hue='Category', data=class_ratings_melted, palette='Spectral')
for c in ax.containers:
        ax.bar_label(c)
customize_plot(ax, 'Mean Ratings across Class', 'Class', 'Mean Rating',13,10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')


# - Travelers in the business class generally give higher average ratings for cleanliness, check-in experience, in-flight wifi, and legroom service.
# - Interestingly, passengers in the business class tend to rate seat comfort comparatively lower.

# In[18]:


corr = customer_df[['Flight Distance', 'Seat comfort', 'Departure/Arrival time convenient',
       'Food and drink', 'Gate location', 'Inflight wifi service',
       'Inflight entertainment', 'Online support', 'Ease of Online booking',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Cleanliness', 'Online boarding',
       'Departure Delay in Minutes', 'Arrival Delay in Minutes']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 10))
sns.heatmap(corr, mask=mask, center=0, linewidths=1, annot=True, fmt=".2f",cmap = 'coolwarm')


# In[19]:


customer_df.drop(columns = ['Arrival Delay in Minutes'],inplace = True)


# # Encoding Categorical Features

# In[20]:


dummies=pd.get_dummies(customer_df['Class'], dtype=int)
dummies


# In[21]:


customer_encoded = pd.concat([customer_df,dummies], axis = 'columns')
customer_encoded.drop(columns = ['Class'], inplace=True)
customer_encoded


# In[22]:


customer_encoded['Customer Type'] = customer_encoded['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
customer_encoded['Type of Travel'] = customer_encoded['Type of Travel'].map({'Personal Travel': 1, 'Business travel': 0})
customer_encoded['satisfaction'] = customer_encoded['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})

customer_encoded


# # Splitting Data

# In[23]:


X = customer_encoded.drop(columns = ['satisfaction'])
y = customer_encoded['satisfaction']


# In[24]:


X.shape, y.shape


# In[25]:


X


# In[26]:


y


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
X_train.shape, y_train.shape


# # Scaling Data

# In[28]:


scaler = StandardScaler()


# In[29]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # Selecting Best Model

# In[30]:


models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'XGB Classifier': XGBClassifier()
}
results = []

for name, model in models.items():
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    print(f'CV Score (Mean) {name}: {np.mean(cv_results)}')
    results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.title('Cross-validation Scores for Classification Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()


# # Random Forest Classifier

# In[31]:


# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions 
y_pred = rf_model.predict(X_test)
# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')
print(classification_report(y_test, y_pred))


# # Model Evaluation

# In[32]:


plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# - The Random Forest Model achieves high precision, recall, and F1-score for both classes, indicating that it performs well in classifying both dissatisfied and satisfied customers.
# - The overall accuracy of 96% suggests that the model is accurate in predicting the customer satisfaction status.

# - Let's try Extreme Gradient Boosting 

# # Extreme Gradient Boosting (XGBoost Classifier)

# In[33]:


# Initialize and train the XGBoost model
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
# Make predictions 
y_pred = xgb_model.predict(X_test)
# Evaluate the model 
accuracy_xgb = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy_xgb * 100:.2f}%') 
print(classification_report(y_test, y_pred))


# ## Model Evaluation

# In[34]:


plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test, y_test, cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# # Conclusion

# - Both Random Forest and XGBoost models exhibit comparable performance metrics, including accuracy, precision, recall, and F1-score.
# - However, the XGBoost model demonstrates a slightly lower number of false positives and false negatives compared to the Random Forest model.
# - This suggests that the XGBoost model outperforms the Random Forest model slightly in terms of minimizing classification errors.
