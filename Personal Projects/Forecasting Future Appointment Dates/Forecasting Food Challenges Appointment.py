#!/usr/bin/env python
# coding: utf-8

# # Appointment Forecasting 

# The purpose of this project is to predict future appointment slots needed per week for each patient going through their treatment cycle. The steps of this project is comprised of 2 sections:
# 
# **Section 1:** Generating a forecast date for each appointment that has not yet been scheduled. 
# 
# **Section 2:** Cross referencing with existing appointment data for cancelled appointments or already scheduled appointments. 
# 
# 

# ## Table of Contents
# 
# * [About the Data](#AboutData)
# * [Section 1: Generating a Forecast Date For Each Treatment Cycle](#Section1)
#     * [Section 1.1: Treatment Cycle Data Clean-Up and Wrangling](#section_1_1)
#     * [Section 1.2: Creating a Forecast Date](#section_1_2)
# * [Section 2: Cross - Referencing Existing Appointments](#Section2)
#     * [Section 2.1: Appointment Data Clean-Up](#Section2.1)
#     * [Section 2.2: Cross Referencing Treatment Cycle Forecast and Appointment Data](#Section2.2)
# 

# ## About the Data <a class="anchor" id="AboutData"></a>

# The datasets used are 
# 1. Patient Treatment Cycle 
# 2. Appointment Data 

# In[4]:


#importing tools needed 
import numpy as np 
import pandas as pd
import datetime 


# In[5]:


#reading the file and converting data types
ap = pd.read_csv('appointments_data.csv')
df = pd.read_csv('patient_treatment_cycle.csv')
df.shape


# #### Appointment Data

# In[6]:


ap.head()


# The appointment data contains information on all historic appointments that a pateint (athenaid) has made. The 'athenaid' and 'appt schdling prvdr' has been encoded to hide information relating to the patient and practice. 
# 
# The appointment month, year, date, startime indicates information relating to a scheduled appointment. 
# 
# | Column| Description  |
# |------|------|
# |apptcancelreason  | if the appointment was cancelled ( indicated in'apptslotstatus')|
# |apptstatus | the status of an appointment  |
# |apptslotstatus|whether the appointment slot was scheduled, cancelled, or open|
# |appttype  | type of appointment |
# |appt schdlng prvdr  | specific provider, encoded by a number|
# |ptnt rmssn | indicates that patient has reached remission  |
# |zn1 | patient resides Western side of U.S.|
# |zn2 |patient resides Eastern side of U.S.|
# |zn3|Paient is not from U.S.|

# #### Treatment Cycle

# In[7]:


df.head(10)


# This dataset contains information of each patient's (athena_id) treatment cycle.
# 
# To understand the data, we'll look at a specific pateint ( 1364 ). For this patient, their appointment date is 10/19/2020 (Treatment Type = Start, Challenge). Once they leave their appointment, they'll have 6 weeks of at-home dosings ( Treatment Type = Weekly x6 ) and after 6 weeks, they'll need an appointment again. 
# 
# Some appointment dates that have alrady occured in the **Pateint Treatment Cycle** will have an existing future appointment in the **Appointment Data**. If the appointment date has not occured, they'll need a forecasted appointment date because there isn't an appointment scheduleld in the **Appointment Data.**

# # Section 1: Generating A Forecast Date For Each Patient Treatment Cycle <a class="anchor" id="Section1"></a>

# In this section, the goal is to forecast an appointment date for each treatment cycle already in effect or scheduled. This date is based off of the weekly at-home dosing cycle. After they've completed their weeks of at home dosing, they'll need an appointment date. 

# ### 1.1: Treatment Cycle Data Cleanup and Wrangling
# <a class="anchor" id="section_1_1"></a>

# In this section we'll read the challenges data and clean up the datafile.

# In[8]:


df.columns


# In[9]:


#Correcting the format for columns
df.columns = ['appointment_date', 'athena_id', 'date_of_birth', 'treatment_type', 'food', 'dose_unit', 'amount', 'num_dose']
df.columns


# After correcting the format of the data, we'll look at the data types to make sure they're accurate. 

# In[10]:


df.dtypes


# This shows that the data types are incorrect for dates. Before correcting them, we'll remove any columns we won't be using. 

# In[11]:


#removing columns that are not used 
df = df.drop(['dose_unit', 'amount', 'date_of_birth', 'num_dose'], axis = 1 )


# In[12]:


#changing data type 
df['appointment_date'] = pd.to_datetime(df['appointment_date'])
df.head(5)


# ### 1.2  Creating a Forecast Date <a class="anchor" id="section_1_2"></a>

# Because this data lists all the appointment dates for all unique id's, we only want to forecast dates of the most current appointment. For each recent appointment date, the number of rows where appointment type is 'weekly', will be added to the date to get forecast date. Note: Weekly counting is based off of food type for each appointment date.

# In[13]:


# grouping AthenaID, Date, Food, Treatment Type and counting by treatment type 
df['weekly_count'] = df.groupby(['athena_id', 'appointment_date', 'treatment_type', 'food'])[['treatment_type']].transform('count')
df.head()


# Because we removed uncessary columns, we'll be able to group by appointment date, athena id. After grouping, there are duplicates, and Null values. We also only want to count the treatment type for "weekly', so we'll remove them. 
# 
# **Observations**
# 1. There are multiple food challenges for at home dosing. 
# 2. The most recent appointment date for each unique appointment id can be an appoinment that has already occured( in which their next appointment date is already in file) OR has not occured ( in which their next appointment date is not scheduled).

# In[14]:


#cleaning up by removing challenges and start, and null values 
df = df[df["treatment_type"] !='Start']
df = df[df["treatment_type"] !='Challenge']
df = df.dropna()


# In[15]:


# adding a column that indicates the number of days to add to the appointment date
df['days_add'] = df['weekly_count'] * 7

# changing datatype of counts into int 
df['weekly_count'] = df['weekly_count'].astype(int)
df['days_add'] = df['days_add'].astype(int)


# Now that we have the necessary information, we can forecast our appointment dates.

# In[16]:


#calculating the forecast date 
df['forecast_date'] = df['appointment_date'] + pd.to_timedelta(df['days_add'], unit= 'd')
df['forecast_date'] = pd.to_datetime(df['forecast_date']).dt.strftime("%Y-%m-%d")


# In[17]:


#dropping duplicates
df = df.drop_duplicates()


# In[18]:


df = df.sort_values( [ 'athena_id','appointment_date','treatment_type' ,'weekly_count'], ascending = [False, False, True, False] ).drop_duplicates(['athena_id', 'treatment_type'])
df


# ## Section 2: Cross - Referencing Exising Appointments <a class="anchor" id="Section2"></a>

# In this section, we'll be taking the forecasted appointment dates from the **Treatment Cycle** and compare with the **Appointment Data** to see if the appointments are accurate for the appointments already scheduled. Next, we'll cross-reference to see if there are any cancelled appointments. 

# ### 2.1: Appointment Data Clean-Up 
# First, import our appointment file from Athena.<a class="anchor" id="Section2.1"></a>

# In[19]:


df.shape


# In[20]:


ap.columns


# In[21]:


#renaming the columns
ap.columns = ['apptmnth', 'apptyear', 'apptdate', 'apptstarttime', 'apptcancelreason',
       'apptstatus', 'apptslotstatus', 'appttype', 'appt schdlng prvdr',
       'athena_id', 'ptnt rmssn', 'zn1', 'zn2', 'zn3']
    


# In[22]:


ap.head()


# Because this data may be useful to create a prediction model, we'll keep all the columns. However, ap.dtypes reveal that data types are wrong. Let's change it for the dates. 

# In[23]:


# correcting the data types
ap['apptdate'] = pd.to_datetime(ap['apptdate']).dt.strftime("%Y-%m-%d")


# ### 2.2: Cross Referencing Appointments 
# <a class="anchor" id="Section2.2"></a>
# For the appointment type, the ones we're looking at to forecast food challenges are 1) Launch Visits 2) Food Challenges( and all categories that have food challenges). 
# 
# For appointments, we'll replace the forecasted appointment with the acutal scheduled appointments if there is one. After, we'll confirm that the appointment is not cancelled. 

# In[24]:


# First remove rows that are irrelevant for this project. This would be any rows were appointment types are not launches or food challenges.
ap = ap[ap['appttype'] != 'Blood Draw']
ap = ap[ap['appttype'] != 'Remission Call']
ap = ap[ap['appttype'] != 'Remission Annual']
ap = ap[ap['appttype'] != 'Remission Lab']
ap = ap[ap['appttype'] != 'Post Remission Food Challenge']
ap = ap[ap['appttype'] != 'Remission Visit 1']
ap = ap[ap['appttype'] != 'Skin Test']
ap = ap[ap['appttype'] != 'Tolerance Visit 2']
ap = ap[ap['appttype'] != 'Tolerance Visit 1']
ap = ap[ap['appttype'] != 'Onboarding Visit']
ap = ap[ap['appttype'] != 'Launch Visit']

# Remove rows labeled open slots
ap = ap[ap['apptslotstatus'] != 'o - Open Slot']


# Observations of the athena data: each athena ID lists all appointment dates for every appointment type every scheduled. We only want the most recent appointment. 

# To make things easier, we'll make a copy of ap with just the columns needed for this. ap is kept because other columns may be needed for exploratory analysis.

# In[25]:


ap_view = ap[['athena_id','apptdate', 'appttype', 'apptslotstatus', 'apptcancelreason']]


# In[26]:


ap_view.head()


# In[27]:


#Merging the df and ap_view files. 
ap_view = pd.merge(df, ap_view, on= 'athena_id', how = 'inner')
ap_view.head()


# If an appointment has already occurred, that means they should have a scheduled next appointment date. So for each df['appointment_date'] < today, replace the forecast date with ap['apptdate'].

# In[28]:


#removing rows with cancelled appointments 
ap_view = ap_view[ap_view['apptslotstatus'] != 'x - Cancelled']
ap_view.dtypes


# In[29]:


# getting todays date
today = datetime.datetime.today()
now = str(today)
now


# In[30]:


#converting the data tyoes
ap_view['appointment_date'] = pd.to_datetime(ap_view['appointment_date']).dt.strftime("%Y-%m-%d")

ap_view['apptdate'] = pd.to_datetime(ap_view['apptdate']).dt.strftime("%Y-%m-%d")

ap_view['forecast_date'] = pd.to_datetime(ap_view['forecast_date']).dt.strftime("%Y-%m-%d")


# In[31]:


ap_view.head()


# Because each treatment cycle can include multiple foods, the food cycle with the most weekly cycle will be taken. 

# In[32]:


# dropping duplicates in the athena_id and apptdate. 
ap_view = ap_view.sort_values( [ 'athena_id','apptdate'], ascending = [False, False] ).drop_duplicates(['athena_id'])
ap_view.head()


# Once we've cleaned up the dataset to show only the most current appointment date, we'll keep only the forecast date if the appointment date is after today. Else, make the forecast date the appointment date. This way, the forecast date column will only show appointment for the future. 

# In[33]:



for index, row in ap_view.iterrows():
    appointment_date = row['appointment_date']
    forecast_date = row['forecast_date']
    apptdate = row['apptdate']
    
    if appointment_date < now:
        ap_view.loc[index,'forecast_date'] = ap_view.loc[index,'apptdate']

ap_view.head()

        


# In[34]:


ap_view['forecast_weeknum'] = pd.to_datetime(ap_view['forecast_date']).dt.strftime("%U")


# In[35]:


ap_view.to_csv('forecast.csv', index = False )


# In[36]:


ap_view.head()


# In[ ]:




