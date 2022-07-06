
"""
Random Forest
 
"""
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("E:\\assignments\\Random Forest\\Company_Data (1).csv")
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
sns.countplot(x ='Urban', data = df_new)
sns.countplot(x ='US', data = df_new)
sns.countplot(x ='ShelveLoc', data = df_new)

Y = df['Sales']
df['Sales'].value_counts()
y_mean = Y.mean()

# Converting Y variables into categorical variables
# (Sales greater than or equal to mean is categorised as High, otherwise Low)

Y1=[]
for i in range(0,400,1):
    if Y.iloc[i,]>=y_mean:
        print('High')
        Y1.append('High')
    else:
        print('Low')
        Y1.append('Low')

Y_new = pd.DataFrame(Y1)
list(Y_new)
Y_new.set_axis(['Target'],axis='columns',inplace=True)
sns.countplot(Y_new['Target'])

# Data Transformation
from sklearn.preprocessing import StandardScaler, LabelEncoder
SS=StandardScaler()
LE=LabelEncoder()
X.iloc[:,:7]=SS.fit_transform(X.iloc[:,:7])
for i in range(7,10,1):
    X.iloc[:,i]=LE.fit_transform(X.iloc[:,i])
print(X)
X.head()

Y_new = LE.fit_transform(Y_new)
Y_new

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y_new,test_size=0.25,stratify=Y_new,random_state=1)
X_train.shape

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.4,n_estimators=500)
model=RFC.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

#Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error
acc = accuracy_score(Y_test, Y_pred)
print(acc)

# To check Train and Test Error

tr_err=[]
t_err=[]
set1=np.arange(0.1,1.1,0.1)
for j in set1:
    RFC=RandomForestClassifier(max_features=j,n_estimators=500)
    model=RFC.fit(X_train,Y_train)
        
    Y_pred_tr=model.predict(X_train)
    Y_pred_te=model.predict(X_test)
        
    tr_err.append(np.sqrt(metrics.mean_squared_error(Y_train,Y_pred_tr)))
    t_err.append(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred_te)))
    

TR_err=np.mean(tr_err)
TE_err=np.mean(t_err)

# Here we got 0 train error as we have fit the model with train data

import matplotlib.pyplot as plt
plt.plot(set1,tr_err,label='Training error')
plt.plot(set1,t_err,label='Test error')
plt.xlabel('No of features')
plt.ylabel('Error')
plt.title('Graph')
plt.show()

"""
Inference : by taking features 0.4 i.e., 40% of the variables we can get highest
            accuracy as 81%












