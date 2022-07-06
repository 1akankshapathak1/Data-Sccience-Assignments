"""
Prepare a classification model using SVM for salary data 

"""
import pandas as pd
import seaborn as sns
data_train=pd.read_csv('E:\\assignments\\SVM\\SalaryData_Train(1).csv')
data_train.shape
list(data_train)
data_train.dtypes

data_test=pd.read_csv('E:\\assignments\\SVM\\SalaryData_Test(1).csv')
data_test.shape
list(data_test)

# Data Transformation
#Seperating X and Y from both train and test datasets
X_train=data_train.iloc[:,0:13]
X_train.dtypes
X_train.hist()

# Visualisation
sns.countplot(X_train['workclass'])
sns.countplot(X_train['education'])
sns.countplot(X_train['maritalstatus'])
sns.countplot(X_train['relationship'])
sns.countplot(X_train['sex'])

#Data standardization
X1=X_train[['age','educationno','capitalgain','capitalloss','hoursperweek']]
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X1_scale=SS.fit_transform(X1)
X1_df=pd.DataFrame(X1_scale)
X1_df.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'],axis=1,inplace=True)
X1_df

#Data Label encoding
X2=X_train[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']]
X2.shape
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,8,1):
    X2.iloc[:,i]=LE.fit_transform(X2.iloc[:,i])
type(X2)
X_train_new=pd.concat([X1_df,X2],axis=1)
Y_train=data_train.iloc[:,13:]
Y_train.dtypes

Y_train_scale=LE.fit_transform(Y_train)
Y_train_df=pd.DataFrame(Y_train_scale)
list(Y_train_df)
Y_train_df.set_axis(['Salary'],axis='columns',inplace=True)
Y_train_df.ndim
Y_train_df['Salary'].ndim
      
X_test=data_test.iloc[:,0:13]
X_test.hist()
#Data standardization
X3=X_test[['age','educationno','capitalgain','capitalloss','hoursperweek']]
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X3_scale=SS.fit_transform(X3)
X3_df=pd.DataFrame(X3_scale)
X3_df.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'],axis=1,inplace=True)
X3_df

#Data Label encoding
X4=X_test[['workclass','education','maritalstatus','occupation','relationship','race','sex','native']]
X4.shape
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for i in range(0,8,1):
    X4.iloc[:,i]=LE.fit_transform(X4.iloc[:,i])
type(X4)
X_test_new=pd.concat([X3_df,X4],axis=1)

Y_test=data_test.iloc[:,13:]
Y_test_scale=LE.fit_transform(Y_test)
Y_test_df=pd.DataFrame(Y_test_scale)
list(Y_test_df)
Y_test_df.set_axis(['Salary'],axis='columns',inplace=True)
Y_test_df.ndim
Y_test_df['Salary'].ndim

#Loading SVC (Linear kernel)
from sklearn.svm import SVC
svl=SVC(kernel='linear').fit(X_train_new,Y_train_df['Salary'])
Y_pred=svl.predict(X_test_new)

#Metrics 
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test_df['Salary'],Y_pred)
print((acc*100).round(3))

#Loading SVC (Radial Bias Function kernel)
from sklearn.svm import SVC
svr=SVC(kernel='rbf').fit(X_train_new,Y_train_df['Salary'])
Y_pred=svr.predict(X_test_new)

#Metrics 
from sklearn.metrics import accuracy_score
acc1=accuracy_score(Y_test_df['Salary'],Y_pred)
print((acc1*100).round(3))

#Loading SVC (Polynomial kernel)
from sklearn.svm import SVC
svp=SVC(kernel='poly').fit(X_train_new,Y_train_df['Salary'])
Y_pred=svr.predict(X_test_new)

#Metrics 
from sklearn.metrics import accuracy_score
acc2=accuracy_score(Y_test_df['Salary'],Y_pred)
print((acc2*100).round(3))

'''
Inference: From the different kernel functions in svm, the accuracies are almost same.
           accuracy = 80.91%
'''
























