#!/usr/bin/env python
# coding: utf-8

# # Predicting Medical Appointment Cancellation and Data Exploration

# The purpose of this project is to predict whether or not a appointment that has been scheduled will be cancelled in the days/months leading up to the appointment. The steps of this project is comprised of 3 sections:
# 
# **Section 1:** Clean Up
# 
# **Section 2:** Data Exploration
# 
# **Section 3:** Appointment Cancellation Prediction

# ## Table of Contents
# 
# * [About the Data](#AboutData)
# * [Section 1: Clean-Up and Data Wrangling](#Section1)
# * [Section 2: Data Exploration](#Section2)
#     * [Section 2.1: Numerical Features](#Section2.1)
#     * [Section 2.2: Categorical Features](#Section2.2)
# * [Section 3: Model Building and Prediction](#Section3)
# * [Section 4: Reflection](#Section4)

# ## About the data <a class="anchor" id="AboutData"></a>

# This dataset contains all appointments scheduled since 2015, with all historical data on appointment types, dates, and whether or not the appointment scheduled was cancelled. A patientid has been encoded, and indicates a unique patient for every ID. The provider is also encoded with a random number, so that each number represents a different provider. 

# In[1]:


# Importing the tools needed
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
import seaborn as sns
sns.set_style("whitegrid")


from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


fc = pd.read_csv('hist_appt_report.csv')
fc.head()


# Information on some columns that are not as straightforward:
# 
# | Column| Description  |
# |------|------|
# |apptcancelreason  | if the appointment was cancelled ( indicated in'apptslotstatus')|
# |apptslotstatus|whether the appointment slot was scheduled, cancelled, or open|
# |appttype  | type of appointment |
# |appt schdlng prvdr  | specific provider for patient, encoded by a number|
# |ptnt rmssn | indicates that patient has reached remission  |
# |zn1 | patient resides Western side of U.S.|
# |zn2 |patient resides Eastern side of U.S.|
# |zn3|Paient is not from U.S.|
# |ptnt cnsnt t cll/text ysn| whether or not the patient agreed to have a call or text reminder |
# |appt inseligibstatus|insurance elligibility|
# |ptnt rmssn| if the patient has reached remission, their remission date is listed|
# 

# ## Section 1: Clean-Up and Data Wrangling <a class="anchor" id="Section1"></a>

# ### 1.1: First Impressions and Clean-Up

# In[3]:


#checking for Null values
fc.isnull().isnull().any()


# In[4]:


# concat. date and time and changing them to datetimevariable.
fc['appt_date_time'] = pd.to_datetime(fc['apptdate'] + ' ' + fc['apptstarttime'])
fc['schedule_date_time'] = pd.to_datetime(fc['apptscheduledate'] + ' ' + fc['apptscheduletime'])
fc['appt_date_time'] = pd.to_datetime(fc['appt_date_time'])

fc['appt_hour'] = fc['appt_date_time'].dt.hour
fc['apptday'] = fc['appt_date_time'].map(lambda x: x.dayofweek)
#appointment week number 
fc['apptweeknum'] = fc['appt_date_time'].dt.week
fc['scheduleweeknum'] = fc['schedule_date_time'].dt.week

fc['apptmnth'] = pd.DatetimeIndex(fc['appt_date_time']).month


# In[5]:


fc.columns


# In[6]:


#reformating column names
fc.columns = ['patientid', 'apptyear', 'apptmnth', 'apptdate', 'apptstarttime',
       'apptday', 'apptcancelreason', 'apptslotstatus', 'appttype',
       'apptscheduledate', 'apptscheduletime', 'prvdr',
       'zn1', 'zn2', 'zn3', 'sex', 'age',
       'age_mnths', 'call', 'text',
       'insurance', 'rmssn', 'appt_date_time',
       'schedule_date_time', 'appt_hour', 'apptweeknum', 'scheduleweeknum']


# In[7]:


fc.dtypes


# In[8]:


#getting rid of data that is no used 
fc = fc[fc["apptslotstatus"] !='o - Open Slot']
# Because there isn't enough data on ages older than 18, and because the clinic does not have kids over 18, we'll remove them.
fc['patientid'] = fc['patientid'].dropna()


# **Observations**
# 1. The appt_date_time is the dat they came in for that *specific index*      
# 2. schedule_date_time is the date they scheduled for appt_date_time for that *specific index*
# 3. patient_last_seen is the date they were last seen for *not specific to index*
# 4. patient_next_appt is the date are are to been seen next for *not sepcific to index*
#         
#         
# Data starts in February 2018. 
#         

# ### 1.2: Recoding the Target Variable and Features

# The target variable is in the apptslotstatus column is is whether the appointment scheduled was cancelled or not. 

# In[9]:


apptcount = fc['apptslotstatus'].value_counts()
apptcount


# **Recoding Target Variable**
# 
# 0 = All appointments not cancelled
# 1= All cancelled appointments

# In[10]:


# Recoding apptslotstatus
fc['apptslotstatus'].replace("x - Cancelled", 1, inplace = True)
fc['apptslotstatus'].replace(["4 - Charge Entered", "f - Filled", "3 - Checked Out", "2 - Checked In"], 0, inplace = True)


# **Recoding Features Variables**
# The zones, call, and text columns should also be changed so it can be used in the model. 
# For any yes or positive values, we'll code as 1, else 0. 

# In[11]:


# Recoding other variables
fc['zn1'].replace("YES", 1, inplace = True)
fc['zn1'].fillna(0, inplace=True)
fc['zn2'].replace("YES", 1, inplace = True)
fc['zn2'].fillna(0, inplace=True)
fc['zn3'].replace("YES", 1, inplace = True)
fc['zn3'].fillna(0, inplace=True)

fc['call'].replace("Y", 1, inplace = True)
fc['call'].replace("N", 0, inplace = True)
fc['call'].fillna(0, inplace=True)
fc['text'].replace("Y", 1, inplace = True)
fc['text'].replace("N", 0, inplace = True)
fc['text'].fillna(0, inplace=True)


# In[12]:


fc.describe()


# Observations:
# 
# Age: 75% percentile is 12 but the age max is 85. Will have to take a closer look and plot into a graph. 

# In[13]:


g = sns.stripplot(data = fc , y = 'age', jitter = True)
g.set_ylim([0 , 30])
#sns.plt.show()


# #### Time between scheduled and actual appointment
# Calcualate the time between the scheduled date and the appoinment date as 'daysbtwn'

# In[14]:


#subtracting the dates in between 
fc['daysbtwn'] = fc['appt_date_time'].sub(fc['schedule_date_time'], axis=0)
# converting that into days
fc['daysbtwn'] = round((fc['daysbtwn'] / np.timedelta64(1, 'D')).abs(), 0)


# In[15]:


g = sns.stripplot(data = fc , y = 'daysbtwn', jitter = True)
g.set_ylim([0 , 130])
#sns.plt.show()


# In[16]:


g = sns.stripplot(data = fc , y = 'appt_hour', jitter = True)
g.set_ylim([0 , 19])
#sns.plt.show()


# Because the data gets sparse with larger daysbetween and age, we'll set cutoffs

# In[17]:


fc = fc[(fc['daysbtwn'] >  1)]
fc = fc[(fc['daysbtwn'] < 150)]

fc = fc[(fc['age'] < 20)]
fc = fc[(fc['age'] > 1)]
fc = fc[(fc['age_mnths'] > 12)]

fc = fc[(fc['appt_hour'] < 17  )]
fc = fc[(fc['appt_hour'] > 7  )]


# #### Number of Appointments Cancelled in the Past 
# Creating the number cancelled, total appointments, and risk scores for each patientid.

# In[18]:


#group by patient id and by cancelled appointments
fc['n_cancelled'] =fc.groupby(['patientid'])['apptslotstatus'].transform('sum')
fc['total_appointment'] = fc.groupby(['patientid'])['apptslotstatus'].transform('count')
fc['risk_score'] = fc['n_cancelled'] / fc['total_appointment']
fc.head()


# In[19]:


g = sns.stripplot(data = fc , y = 'risk_score', jitter = True)
g.set_ylim([0 , 1])


# In[20]:


g = sns.stripplot(data = fc , y = 'n_cancelled', jitter = True)
g.set_ylim([0 , 40])
fc = fc[(fc['n_cancelled'] < 40  )]


# In[21]:


g = sns.stripplot(data = fc , y = 'total_appointment', jitter = True)
g.set_ylim([0 , 50])


# ## Section 2: Exploring the Data <a class="anchor" id="Section2"></a>

# ### Exploratory Analysis on Features

# The appttypes need to be cleaned because there seems to be variations of the same appttype. I decided to narrow them down to *Food Challenge*, *Onboarding*, and *remssion* to make things more simple.

# In[22]:


# Replacing Food Challenge variations to just call it food challenge
fc['appttype'].replace("Food Challenge Zone 1 (1c)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1 (1c1i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1 (1c2i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1 (2c)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1 (2c1i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1(1i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1(2i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1 (2C)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1(3C)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 1(3i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (1c)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (1c1i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (1c2i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (1i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (2c)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (2c1i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (2i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (3c)", "Food Challenge", inplace = True)
fc['appttype'].replace("Food Challenge Zone 2/3 (3i)", "Food Challenge", inplace = True)
fc['appttype'].replace("Tolerance Visit 1", "Food Challenge", inplace = True)
fc['appttype'].replace("Tolerance Visit 2", "Food Challenge", inplace = True)
fc['appttype'].replace("Zone 2/Zone 3", "Food Challenge", inplace = True)
fc['appttype'].replace("Launch Visit", "Food Challenge", inplace = True)
fc['appttype'].replace("Repeat Challenge", "Food Challenge", inplace = True)

fc['appttype'].replace("Skin Test", "Onboarding", inplace = True)
fc['appttype'].replace("Blood Draw","Onboarding", inplace = True)
fc['appttype'].replace("Patch Placement", "Onboarding", inplace = True)
fc['appttype'].replace("Pulmonary Follow Up", "Onboarding", inplace = True)

fc['appttype'].replace("Onboarding Visit", "Onboarding", inplace = True)
fc['appttype'].replace("Onboarding Lab", "Onboarding", inplace = True)


fc['appttype'].replace("Post Remission Food Challenge", "Remission", inplace = True)
fc['appttype'].replace("Remission Call", "Remission", inplace = True)
fc['appttype'].replace("Remission Visit 1", "Remission", inplace = True)
fc['appttype'].replace("Remission Annual", "Remission", inplace = True)
fc['appttype'].replace("Remission Lab", "Remission", inplace = True)


fc['apptcancelreason'].fillna("Filled", inplace = True)


# Next, we'll see how each features weigh in on the apptslotstatus being cancelled. 

# In[23]:


# finding the average cancellations
fc['apptslotstatus'].mean()


# In[24]:


# finding out how each feature weighs in on the apptslotstatus being cancelled
features = [ 'appttype','apptmnth','apptweeknum', 'appt_hour', 'apptday', 'zn1', 'zn2', 'zn3', 'age', 
       'call', 'text', 'daysbtwn', 'appt_hour','prvdr', 'risk_score', 'apptcancelreason', 'insurance', 'rmssn']
for i in features:
    print(fc.groupby(i)['apptslotstatus'].mean())


# ### 2.1: Numerical Features <a class="anchor" id="Section2.1"></a>

# In[25]:


def prob( x, y):
    df = pd.crosstab( index = x[y] , columns = x['apptslotstatus']).reset_index()
    df['fulfilled'] = df[0] / (df[1] + df[0])
    return df[[y, 'fulfilled']]


# In[26]:


sns.lmplot( x = 'age', y = 'fulfilled',  data = prob(fc, 'age'), fit_reg = True)
sns.lmplot( x = 'age_mnths', y = 'fulfilled', data = prob(fc, 'age_mnths'), fit_reg = True)
sns.lmplot( x = 'daysbtwn', y = 'fulfilled', data = prob(fc, 'daysbtwn'), fit_reg = True)
sns.lmplot( x = 'apptday', y = 'fulfilled', data = prob(fc, 'apptday'), fit_reg = True)
sns.lmplot( x = 'appt_hour', y = 'fulfilled', data = prob(fc, 'appt_hour'), fit_reg = True)
sns.lmplot( x = 'apptweeknum', y = 'fulfilled', data = prob(fc, 'apptweeknum'), fit_reg = True)
sns.lmplot( x = 'apptmnth', y = 'fulfilled', data = prob(fc, 'apptmnth'), fit_reg = True)


# In[27]:


#fc = fc[(fc['daysbtwn'] < 100  )]
fc = fc[(fc['daysbtwn'] != 32  )]
fc = fc[(fc['daysbtwn'] != 119  )]

fc = fc[(fc['apptweeknum'] != 119  )]

fc = fc[(fc['appt_hour'] != 16)]


# In[28]:


sns.lmplot( x = 'n_cancelled', y = 'fulfilled', data = prob(fc,'n_cancelled'), fit_reg = True)
sns.lmplot( x = 'risk_score', y = 'fulfilled', data = prob(fc,'risk_score'), fit_reg = True)


# In[29]:


plt.hist(fc['age_mnths'])
plt.title('Age')
plt.show()
plt.hist(fc['appt_hour'])
plt.title('Appt_Hour')
plt.show()
plt.hist(fc['daysbtwn'])
plt.title('Days Between')
plt.show()
plt.hist(fc['apptweeknum'])
plt.title('Week Number')
plt.show()


# Possible outliers in days between. Let's work on identifying and removing them. 

# ### 2.2: Categorical Features <a class="anchor" id="Section2.2"></a>

# In[30]:


def probcat(group_by):
    rows = []
    for item in group_by:
        for level in fc[item].unique():
            row = {'Condition': item}
            total = len(fc[fc[item] == level])
            n = len(fc[(fc[item] == level) & (fc['apptslotstatus'] == 0)])
            row.update({'Level': level, 'Probability': n / total})
            rows.append(row)
        return pd.DataFrame(rows)


# In[31]:


#probability of showing up to appointment 
sns.barplot(data = probcat(['call']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['text']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['zn3']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['zn2']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['zn1']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['appttype']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set3')
plt.show()
sns.barplot(data = probcat(['apptday']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['prvdr']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()
sns.barplot(data = probcat(['insurance']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()

sns.barplot(data = probcat(['apptmnth']), x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')
plt.show()


# In[32]:


fc = fc[(fc['apptmnth'] != 3)]
fc = fc[(fc['apptmnth'] != 4)]


# In[33]:


#encoding appointment type 
fc['appttype_encoded'] = LabelEncoder().fit_transform(fc['appttype'])
fc['ins_encoded'] = LabelEncoder().fit_transform(fc['insurance'])
fc['can_encoded'] = LabelEncoder().fit_transform(fc['apptcancelreason'])


# ## Part 3: Modeling Building <a class="anchor" id="Section3"></a>

# ### Training Set

# We'll use a train dataset to fir the ml model and test dataset to evaluate the fir machine learning model. 

# In[34]:


#training set
X = fc[['apptyear','apptmnth','appt_hour','appttype_encoded','apptday','apptweeknum', 'zn1', 'zn2', 'zn3', 'call', 'risk_score',  'ins_encoded', 'scheduleweeknum','daysbtwn']]
#X = fc[[ 'risk_score', 'n_cancelled']]

#X = fc[['apptyear','apptmnth', 'appt_hour','appttype_encoded','apptday','apptweeknum', 'zn1', 'zn2', 'zn3', 'age','age_mnths', 'call', 'text', 'risk_score', 'n_cancelled', 'total_appointment', 'ins_encoded', 'prvdr_encoded','scheduleweeknum','daysbtwn', 'id_encoded']]
y = fc['apptslotstatus']


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20)
#from sklearn.model_selection import TimeSeriesSplit

#tss = TimeSeriesSplit(n_splits = 3 )

#for train_index, test_index in tss.split(X):
#    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
#    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# ### Prediction

# In[36]:


gnb = GaussianNB()
cv = cross_val_score(gnb,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[37]:


lr = LogisticRegression(max_iter = 53000)
cv = cross_val_score(lr,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[38]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[39]:


dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[40]:


knn = KNeighborsClassifier()
cv = cross_val_score(knn,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[41]:


rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf,X_train,y_train,cv=5)
print(cv)
print(cv.mean())


# In[42]:


from sklearn.metrics import mean_absolute_error
rf.fit(X_train, y_train)
yhat = rf.predict(X_test).astype(int)

mae= mean_absolute_error(y_test, yhat)
print ( 'MAE: %3f' %mae)
# MAE < 10% excellent 
# MAE < 20% Good 


# In[43]:


#Predicting on entire dataset
yhat2 = rf.predict(X)
fc['yhat2'] = yhat2


# In[44]:


fc.to_csv('cancellation_prediction.csv', index = False)


# In[45]:


fc.head(50)


# ## Reflection <a class="anchor" id="Section4"></a>

# The Random Forest model yielded an accuracy of 75%. Some limitations of this dataset is the lack of consistancy in the appttypes. There were many apptypes that had to be narrowed down into categories and may have been too generalized on the weight of cancelled appointments. I would also note that I removed dates in March, June, and July since COVID-19 effected the cancellations, it would also have effected the accuracy. It's possible that the appointment cancellation predictions would be more accurate had times been more normal. 

# In[ ]:




