
"""
A cloth manufacturing company is interested to know about the segment 
or attributes causes high sale. 
"""
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("Company_Data.csv")
df
type(df)
df.shape
df.ndim
df.head()
df.tail()
df.isnull().sum()
df.describe()
df.corr()
X1=df['ShelveLoc']
df.drop(['ShelveLoc'],axis=1,inplace=True)
df_new=pd.concat([df,X1],axis=1)
df_new
X=df_new.iloc[:,1:11]
X.shape
list(X.iloc[:,:7])
X.iloc[:,:7].hist()
sns.distplot(X.iloc[:,:7])

Y = df['Sales']
df['Sales'].value_counts()

Y1=[]
# Converting Y variables into categorical variables
for i in range(0,400,1):
    if Y.iloc[i,]>=Y.mean():
        print('High')
        Y1.append('High')
    else:
        print('Low')
        Y1.append('Low')
Y_new=pd.DataFrame(Y1)

#Preprocessing the data
from sklearn.preprocessing import StandardScaler, LabelEncoder
SS=StandardScaler()
LE=LabelEncoder()
X.iloc[:,:7]=SS.fit_transform(X.iloc[:,:7])
for i in range(7,10,1):
    X.iloc[:,i]=LE.fit_transform(X.iloc[:,i])
print(X)
X.head()

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y_new,test_size=0.25,stratify=Y_new,random_state=91)
X_train.shape

#Fitting the model
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(criterion='entropy',max_depth=8).fit(X_train,Y_train)
Y_pred=DT.predict(X_test)

#Metrics
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print(acc)

#Tree
import matplotlib.pyplot as plt
from sklearn import tree
tr=tree.plot_tree(DT,filled=True,fontsize=6)

DT.tree_.node_count
DT.tree_.max_depth
'''
Inference: For random state 91, max depth of 8 in DT  
           accuracy = 81%
'''




















