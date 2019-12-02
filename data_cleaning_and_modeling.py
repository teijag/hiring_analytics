#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Package dependencies
import pandas as pd
import numpy as np
import seaborn
import datetime 
from openpyxl import load_workbook

from matplotlib import pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[27]:


# helper funtions
def catFowler(x):
    if (x == "Fowler White Boggs"):
        return 1
    else:
        return 0
def regroup_section(x,major_sections = ["Litigation", "FIBRE","LEBI","Tax","Intellectual Property","State Gov't Relations"]):
    if (x not in major_sections):
        x = "Other"
    return x

def regroup_office(x,major_offices = ["TMP", "PHIL", "DC", "NYC", "FLL", "PGHO", "FTM"]):
    if (x not in major_offices):
        x = "Other"
    return x

def group(x):
    if (x == "x"):
        return 1
    else:
        return 0
def bonus(x,d1):
    if (x <= d1):
        return 0
    else:
        return 1
def roc(x):
    if (x > 0):
        return 1
    else:
        return 0
    
def Working(x):
    if (x > 0):
        return 2
    else:
        return 0
    
def categorize_firm_size(firm_size):
    if firm_size >=700:
        return ">=700"
    elif firm_size >=100:
        return ">=100"
    else:
        return "<100"
    
def add_previous_firm_size_feature(df):
    df["previous_firm_size"] = df["Number of Lawyers at Previous Firm"].apply(lambda x:categorize_firm_size(x))
    df = df.drop(columns = ["Number of Lawyers at Previous Firm"])
    return df
def feature_importance(clf,df_ml):
    if ('Months to Reach Proforma Expectation' in df_ml.columns):
        col_list = df_ml.drop(columns=['Months to Reach Proforma Expectation']).columns
    else:
        col_list = df_ml.columns
    rank = np.argsort(-np.abs(clf.coef_[0]))
    for r in rank:
        print (col_list [r],clf.coef_[0][r])
    


# ### 1. Data Cleaning and Feature Selection

# In[28]:


# read excel file to panda dataframe
df = pd.read_excel("Lateral Hire Analysis.xlsx",sheet_name = "Main Sheet", header = 2)
# filter out rows without proforma expectations or timekeeper number
df = df[df['Timekeeper Number'].notnull()]
proforma_filter = (df["Proforma Expectations WORKING"]> 0) | (df["Proforma Expectations ROC"]> 0)
df = df[proforma_filter]

# choose relevant columns for data cleaning
dfToClean = df[[
            'Timekeeper Number',
            'Set',
            'Proforma Hire Level', 
            'Group',
            'Office', 
            'Section ',
            'Hire Date',
            'Number of Lawyers at Previous Firm', 
            'Previous Employer',
            'Rehire (yes/no)', 
            'Length of Employment at Previous Firm/Position', 
            'Maximum Length of Employment at any Previous Firm', 
            'Total Years of Experience at Time of Hire', 
            'Internal Referral (yes/no)', 
            'Headhunter (yes/no)',
            'Proforma Base Compensation (starting salary at Buchanan)', 
            'Bonus Component to Base Compensation Based on ROC Revenues (yes/no)', 
            'Proforma Expectations ROC', 
            'Proforma Expectations WORKING', 
            'Due Diligence Projections', 
            '# of Clients Expected to Port', 
            'Previous Year (Y3)', 
            'Previous Year (Y2)', 
            'Previous Year (Y1)', 
            'Previous Firm Standard Rate', 
            'BIR Proforma Rate (starting Standard Rate at Buchanan)',
           'Months to Reach Proforma Expectation']]

# drop original index columns
dfToClean.reset_index(inplace=True)
dfToClean.drop(['level_0', 'index'], axis=1, inplace=True, errors='ignore')

# clean and type cast 3 previous year working hours columns
dfToClean['Previous Year (Y3)'].replace("-",0, inplace=True)
dfToClean['Previous Year (Y3)'].fillna(0, inplace=True)
dfToClean['Previous Year (Y3)'] = dfToClean['Previous Year (Y3)'].astype(float)

# clean and type cast previous firm standard rate
dfToClean['Previous Firm Standard Rate'].replace(["#N/A", "n/a", "not prov."],0, inplace=True)
dfToClean['Previous Firm Standard Rate'].fillna(0, inplace=True)
dfToClean['Previous Firm Standard Rate'] = dfToClean['Previous Firm Standard Rate'].astype(float)

# adding new binary feature - fowler vs non-fowler. We only look at non-fowler people
fowler_series = dfToClean["Previous Employer"].apply(lambda x: catFowler(x))
dfToClean["Fowler"] = fowler_series

# adding new binary feature - group vs solo
group_series = dfToClean["Group"].apply(lambda x: group(x))
dfToClean["GroupOrNot"] = group_series

# adding new binary feature - new bonus structure applied after 2/12/2018
d1 = datetime.datetime(2018, 2, 12) 
bonus_series = dfToClean["Hire Date"].apply(lambda x: bonus(x,d1))
dfToClean["New_Bonus_Structure"] = bonus_series

# adding new binary feature - whether attorney is roc or working
roc_series = dfToClean["Proforma Expectations ROC"].apply(lambda x: roc(x))
working_series = dfToClean["Proforma Expectations WORKING"].apply(lambda x: Working(x))
dfToClean['roc']=roc_series
dfToClean['working']=working_series
dfToClean["rocVsWorking"] = dfToClean['roc'] +dfToClean['working']

# adding new categorical features "previous_firm_size" feature
dfToClean = add_previous_firm_size_feature(dfToClean)

# These columns are for identification purposes.
miscColumns = ['Set','Hire Date','Timekeeper Number']

# These columns are numeric columns to be type casted
numColumns = [
            'Length of Employment at Previous Firm/Position', 
            'Maximum Length of Employment at any Previous Firm', 
            'Total Years of Experience at Time of Hire', 
            'Proforma Base Compensation (starting salary at Buchanan)', 
            'Proforma Expectations ROC', 
            'Proforma Expectations WORKING', 
            'Due Diligence Projections', 
            '# of Clients Expected to Port', 
            'Previous Year (Y3)', 
            'Previous Year (Y2)', 
            'Previous Year (Y1)', 
            'Previous Firm Standard Rate', 
            'BIR Proforma Rate (starting Standard Rate at Buchanan)']

# These columns are binary columns to be type casted
binaryColumns = ['Rehire (yes/no)',
             'Internal Referral (yes/no)', 
             'Headhunter (yes/no)',
             'Bonus Component to Base Compensation Based on ROC Revenues (yes/no)',
            'Fowler',
             'New_Bonus_Structure',
             'GroupOrNot',
            'Months to Reach Proforma Expectation']

# These columns are categorical columns to be type casted
catColumns = ["previous_firm_size",
              'Office_cleaned',
              'Proforma Hire Level',
              'Section_cleaned',
              'rocVsWorking']

# regroup sections and offices into the major ones 
dfToClean["Section_cleaned"] = dfToClean["Section "].apply(lambda x: regroup_section(x))
dfToClean["Office_cleaned"] = dfToClean["Office"].apply(lambda x: regroup_office(x))

# typecasting
binaryDf = pd.DataFrame()
for i in binaryColumns:
    if (i == "Months to Reach Proforma Expectation"):
        binaryDf[i] = np.where(dfToClean[i]<=16, 1, 0)
    elif (i not in ["Fowler","GroupOrNot","New_Bonus_Structure"]): #Fowler, GroupOrNot and NewBonusStructure are already in the correct format
        binaryDf[i] = np.where(dfToClean[i]=='Y', 1, 0)
    else:
        binaryDf[i] = dfToClean[i]
encodedDf = pd.DataFrame()
for i in catColumns:
    dummies = pd.get_dummies(dfToClean[i], prefix=i)
    encodedDf = pd.concat([encodedDf, dummies], axis=1)

# concat all types of columns
numericDf = dfToClean[numColumns]
master = pd.concat([numericDf, binaryDf, encodedDf,dfToClean[miscColumns]], axis=1)
master.fillna(0, inplace=True)

# drop fowler people
master = master[master["Fowler"]==0]    
# Split data into model data and prediction data
train_df = master[master["Set"] == "Model"]
test_df = master[master["Set"] == "Predict"]

# Drop columns that will not be used in modelling and prediction
train_df = train_df.drop(columns = ["Hire Date","Fowler","Set"])
test_df = test_df.drop(columns = ["Hire Date","Fowler","Set"]) 


# ### 2. Model training and reporting

# In[29]:


# drop identifiers
train_df = train_df.drop('Timekeeper Number', axis=1)


# In[30]:


# split data into input features and target
X = train_df.drop(columns = ['Months to Reach Proforma Expectation']).values
y = train_df['Months to Reach Proforma Expectation']

# split data into train and validation data set
X_train, X_val, y_train, y_val = train_test_split(X, y)

# scale the data
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_val_scale = scaler.transform(X_val)

# fit the model and predict
clf = LogisticRegression(random_state=0, max_iter = 1000).fit(X_train_scale, y_train)
y_predict = clf.predict(X_val_scale)

# report accuracy score and feature importance of the model
print ("model acc score on validation set:", accuracy_score(y_val,y_predict))
print ("feature importance")
feature_importance(clf,train_df)


# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif["features"] = train_df.drop(columns = ['Months to Reach Proforma Expectation']).columns


# In[32]:


vif


# In[33]:


X = train_df.drop(columns = ['Months to Reach Proforma Expectation'])
columns = X.columns.tolist()
threshold = 5
dropped = True
remain_cols = columns.copy()
while dropped:
    dropped = False
    vif = [variance_inflation_factor(X[remain_cols].values, ix) for ix in range(len(remain_cols))]
    print (vif)
    max_vif = vif.index(max(vif))
    if (max(vif) > threshold):
        print('dropping \'' + remain_cols[max_vif] + '\' at index: ' + str(max_vif))
        remain_cols.pop(max_vif)
        #remain_cols = remain_cols.drop(X.columns[max_vif])
        dropped=True
print('Remaining variables:')
print(remain_cols)

fe_X = X[remain_cols]


# In[34]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(fe_X.values, i) for i in range(len(remain_cols))]
vif["features"] = remain_cols


# In[35]:


vif


# In[36]:


# split data into input features and target
X = fe_X.values
y = train_df['Months to Reach Proforma Expectation']

# split data into train and validation data set
X_train, X_val, y_train, y_val = train_test_split(X, y,test_size = 0.2)

# scale the data
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_val_scale = scaler.transform(X_val)

# fit the model and predict
clf = LogisticRegressionCV(cv = 10, max_iter = 1000).fit(X_train_scale, y_train)
y_predict = clf.predict(X_val_scale)

# report accuracy score and feature importance of the model
print ("model training score", clf.score(X_train_scale,y_train))
print ("model acc score on validation set:", accuracy_score(y_val,y_predict))
print ("feature importance")

feature_importance(clf,fe_X)


# ### 3. Predict results on new spreadsheet 

# In[37]:


# store identifiers for displaying purpose before dropping it 
timekeeper = test_df['Timekeeper Number']

# select features 
X = test_df[remain_cols]

# scale data
X_scale = scaler.transform(X)

# predict a binary outcome and the probability
y_pred = clf.predict(X_scale)
y_prob = clf.predict_proba(X_scale)[:,1]

# build a new table and write to a new excel file
output_df = pd.DataFrame()
output_df["Timekeeper Number"] = timekeeper
output_df["Probability of Success"] = y_prob
output_df["Predicted Success"] = y_pred

output_df.to_csv("prediction_result.csv",index=False)


# In[ ]:




