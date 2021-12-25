# Logistic-Regression

# importing required libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# reading test and train data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# Dividing dependent and independent variables
train_x = train.drop('subscribed', axis=1)
train_y = train['subscribed']

# Train_x variable Head
train_x.head()

# removed variables that doesn't affect results
train_x = train_x.drop('month', axis=1)
train_x = train_x.drop('day', axis=1)
train_x = train_x.drop('ID', axis=1)


# In[121]:

# Using describe function to glance over distribution of data
train_x.describe()

# Replacing categorical variables into discrete variables
# Using Map function to map Jops with respective level of heirarachy based on fact that people having better job tend to subscribe more.
job_mapping = {'management': 3, 'technician': 2, 'blue-collar': 2, 'admin.': 2, 'entrepreneur': 1,
               'housemaid': 1, 'retired': 1, 'self-employed': 1, 'services': 1, 'student': 1, 'unemployed': 1, 'unknown': 0.5}
train_x['job'] = train_x['job'].map(job_mapping)

# Mapping Marital Variable
marital_mapping = {'married': 2, 'single': 1.8, 'divorced': 1}
train_x['marital'] = train_x['marital'].map(marital_mapping)

# Mapping Education variable
ed_mapping = {'tertiary': 3, 'secondary': 2, 'primary': 1, 'unknown': 1}
train_x['education'] = train_x['education'].map(ed_mapping)

# Mapping Default Variable
def_mapping = {'no': 1, 'yes': 0}
train_x['default'] = train_x['default'].map(def_mapping)

# Mapping Housing Variable
hou_mapping = {'no': 0.5, 'yes': 0.5}
train_x['housing'] = train_x['housing'].map(hou_mapping)

# Mapping loan Variable
loan_mapping = {'no': 0.6, 'yes': 0.4}
train_x['loan'] = train_x['loan'].map(loan_mapping)


train_x.head()

# removing outliers from duration
# variance and standard deviation of duration variable
train_x['duration'].var(ddof=0)
train_x['duration'].std(ddof=0)
# visualising duration variable using histogram
train_x['duration'].plot.hist()
train_x['duration'].plot.hist(bins=1000)

# Removing Outliers from Duration variable
# IQR of duration variable
IQR_dur = train_x['duration'].quantile(0.75)-train_x['duration'].quantile(0.25)
IQR_dur
# left limit of duration
left_dur = train_x['duration'].quantile(0.25)-1.5*(IQR_dur)
# right limit of duration
rig_dur = train_x['duration'].quantile(0.75)+1.5*(IQR_dur)
# Replacing Outliers with Mean Value
train_x.loc[train_x['duration'] > 640.5,
            'duration'] = np.mean(train_x['duration'])
train_x.head()

# Drawing relation between campaign variable and Dependent Variable
train_x['campaign'].value_counts()

pd.crosstab(train['campaign'], train['subscribed'])

pd.crosstab(train['campaign'], train['subscribed'])/len(train['subscribed'])

# Removing Outliers from campaign variable
left_cam = train_x['campaign'].quantile(
    0.25)-1.5*(train_x['campaign'].quantile(0.75)-train_x['campaign'].quantile(0.25))
right_cam = train_x['campaign'].quantile(
    0.75)+1.5*(train_x['campaign'].quantile(0.75)-train_x['campaign'].quantile(0.25))

# replacing outliers with Mean of campaign variable
train_x.loc[train_x['campaign'] > 6, 'duration'] = np.mean(train_x['campaign'])


train_x.describe()

# Drawing a elation between pday variable
pd.crosstab(train['pdays'], train['subscribed'])/len(train['subscribed'])

# removing pdays as it has no effect on the subscribed variable
train_x = train_x.drop('pdays', axis=1)

# Mapping previous Outcome variable

pout_mapping = {'success': 1, 'unknown': 0, 'failure': -1}
train_x['poutcome'] = train_x['poutcome'].map(pout_mapping)

train_x.head()

# Creating dummies for variables that are equally affecting dependent variable

train_x = pd.get_dummies(train_x)

train_x.head()

# Replacing NA values with 0
train_x.fillna(0, inplace=True)


# Replacing "yes" with 1 and "no" with 0 in dependent Variable Column
train_y_mapping = {'yes': 1, 'no': 0}
train_y = train_y.map(train_y_mapping)


# Removing independent variables that doesn't affect dependent variables from Test Data

test_x = test.drop('month', axis=1)
test_x = test.drop('day', axis=1)
test_x = test_x.drop('pdays', axis=1)
test_x = test_x.drop('ID', axis=1)


# Removing Outliers and replacing with Mean of the Variables in Test Data
IQR_dur_test = test_x['duration'].quantile(
    0.75)-test_x['duration'].quantile(0.25)
left_dur_test = test_x['duration'].quantile(0.25)-1.5*(IQR_dur_test)
rig_dur_test = test_x['duration'].quantile(0.75)+1.5*(IQR_dur_test)
test_x.loc[test_x['duration'] > rig_dur_test,
           'duration'] = np.mean(test_x['duration'])

right_cam_test = test_x['campaign'].quantile(
    0.75)+1.5*(test_x['campaign'].quantile(0.75)-test_x['campaign'].quantile(0.25))
test_x.loc[test_x['campaign'] > 6, 'duration'] = np.mean(test_x['campaign'])


left_bal_test = test_x['balance'].quantile(
    0.25)-1.5*(test_x['balance'].quantile(0.75)-test_x['balance'].quantile(0.25))
rig_bal_test = test_x['balance'].quantile(
    0.75)+1.5*(test_x['balance'].quantile(0.75)-test_x['balance'].quantile(0.25))

test_x.loc[test_x['balance'] > rig_bal_test,
           'duration'] = np.mean(test_x['balance'])
test_x.loc[test_x['balance'] < left_bal_test,
           'duration'] = np.mean(test_x['balance'])


test_x.head()


# In[162]:


test_x.head()


# In[163]:


# replaced Categorical variables with discrete variables in Test Data Set

job_mapping_test = {'management': 3, 'technician': 2, 'blue-collar': 2, 'admin.': 2, 'entrepreneur': 1,
                    'housemaid': 1, 'retired': 1, 'self-employed': 1, 'services': 1, 'student': 1, 'unemployed': 1, 'unknown': 0.5}
test_x['job'] = test_x['job'].map(job_mapping_test)

marital_mapping = {'married': 2, 'single': 1.8, 'divorced': 1}
test_x['marital'] = test_x['marital'].map(marital_mapping)

ed_mapping = {'tertiary': 3, 'secondary': 2, 'primary': 1, 'unknown': 1}
test_x['education'] = test_x['education'].map(ed_mapping)

def_mapping = {'no': 1, 'yes': 0}
test_x['default'] = test_x['default'].map(def_mapping)

hou_mapping = {'no': 0.5, 'yes': 0.5}
test_x['housing'] = test_x['housing'].map(hou_mapping)

loan_mapping = {'no': 0.6, 'yes': 0.4}
test_x['loan'] = test_x['loan'].map(loan_mapping)

pout_mapping_test = {'success': 1, 'unknown': 0, 'failure': -1}
test_x['poutcome'] = test_x['poutcome'].map(pout_mapping_test)

# Filling NAN with 0 in Test Data
test_x.fillna(0, inplace=True)

test_x.head()

# Getting dummies for test data

test_x = pd.get_dummies(test_x)


# Import LogisticRegression from Sk learn

logreg = LogisticRegression(max_iter=1000)

# Training Model
logreg.fit(train_x, train_y)
# predictions
pred = logreg.predict(test_x)
pred

# writing predictions into csv File
fullmodel = pd.DataFrame()
fullmodel['subs'] = pred
fullmodel.to_csv('fullmodel.csv', header=True, index=False)
