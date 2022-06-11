"""
 Use decision trees to prepare a model on fraud data 
treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
"""

import pandas as pd
import numpy as np
df = pd.read_csv("Fraud_check.csv") 
df
df.rename(columns={'Taxable.Income': 'taxable_income'}, inplace=True)
df['taxable_income'].value_counts()
df.loc[df['taxable_income'].idxmax()]
category = pd.cut(df.taxable_income,bins=[0,30000,99619],labels=['Risky','Good'])
df.insert(6,'Income',category)
df1 = df.drop(['taxable_income'], axis=1)

# EDA
type(df1)
df1.shape
df1.ndim
df1.head()
df1.tail()
list(df1)
df1.info()
df1.isnull()
df1.isnull().sum()
df1.describe()
df1.corr()


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(df['Income'])

sns.heatmap(df.isnull(), yticklabels=False, cbar=True)


             
# Transformations
             
from sklearn.preprocessing import LabelEncoder             
LE = LabelEncoder() 
df1["Undergrad_n"] = LE.fit_transform(df1["Undergrad"])
df1["Marital_status_n"] = LE.fit_transform(df1["Marital.Status"])
df1["Urban_n"] = LE.fit_transform(df1["Urban"])


from sklearn.preprocessing import StandardScaler
std=StandardScaler()
col_names = ['City.Population', 'Work.Experience']
features = df1[col_names]
X_scale = std.fit_transform(features.values)
scaled_features = pd.DataFrame(X_scale, columns = col_names)

df2 = pd.concat([df1,scaled_features], axis=1)
df2.head()
list(df2)


# Split X and Y
X = df2.iloc[:,6:]
Y = df2['Income']

#Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=99)

# Fitting the model
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy',max_depth=8)
DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)

# Calculations
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
cm
ac = accuracy_score(Y_test,Y_pred)
ac.round(3)


#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)

DT.tree_.node_count
DT.tree_.max_depth
'''
Inference: For random state 91, max depth of 8 in DT  
           accuracy = 77.2%
'''
































