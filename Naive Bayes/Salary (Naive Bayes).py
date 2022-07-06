"""
 Prepare a classification model using Naive Bayes 
for salary data
"""
import pandas as pd
import numpy as np
import seaborn as sns
data_train = pd.read_csv("E:\\assignments\\Naive Bayes\\SalaryData_Train.csv")
data_test = pd.read_csv("E:\\assignments\\Naive Bayes\\SalaryData_Test.csv")

# Mapping Y variables
Y_train = data_train["Salary"]
Y_test = data_test["Salary"]
np.sort(Y_train)
np.sort(Y_test)
mapping = {' >50K': 1, ' <=50K': 0}
Y_train_N = data_train.replace({"Salary":mapping})
Y_train_new = Y_train_N["Salary"]
Y_test_N = data_test.replace({"Salary":mapping})
Y_test_new = Y_test_N["Salary"]

# EDA for train data
data_train.shape
data_train.ndim
type(data_train)
list(data_train)
data_train.head()
data_train.info()
data_train.isnull().sum()
data_train.describe()
data_train.corr()

# EDA for test data
data_train.shape
data_train.ndim
type(data_train)
list(data_train)
data_train.head()
data_train.info()
data_train.isnull().sum()
data_train.describe()
data_train.corr()

# Visualization of train data
sns.pairplot(data_train)
sns.countplot(x="Salary", data=data_train)
sns.countplot(x="maritalstatus", data=data_train)
sns.countplot(x="sex", data=data_train)
sns.countplot(x="workclass", data=data_train)
sns.countplot(x="native", data=data_train)
sns.heatmap(data_train.isnull(), yticklabels=False, cbar=False)

# Splitting of Data

# Splitting of train data
X_train = data_train.iloc[:,:13]
X_train
list(X_train)
X_train.shape
X_train.ndim
X_train_nu = data_train[['age','educationno','capitalgain','capitalloss','hoursperweek',]]
X_train_nu.hist()

X_train_var = data_train[['workclass','education','maritalstatus','occupation',
'relationship','race','sex','native']]

# Splitting of test data
X_test = data_test.iloc[:,:13]
X_test
list(X_test)
X_test.shape
X_test.ndim
X_test_nu = data_test[['age','educationno','capitalgain','capitalloss','hoursperweek',]]
X_test_nu.hist()

X_test_var = data_test[['workclass','education','maritalstatus','occupation',
'relationship','race','sex','native']]

# Transformation of Train and Test Data
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for i in range(0,8,1):
    X_train_var.iloc[:,i]=LE.fit_transform(X_train_var.iloc[:,i])
    X_test_var.iloc[:,i]=LE.fit_transform(X_test_var.iloc[:,i])
list(X_train_var)
X_train_var.shape
X_train_var.ndim

list(X_test_var)
X_test_var.shape
X_test_var.ndim

from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
X_train_scale=MMS.fit_transform(X_train_nu)
X_test_scale=MMS.fit_transform(X_test_nu)
X_train_NN = pd.DataFrame(X_train_scale)
X_test_NN = pd.DataFrame(X_test_scale)
df2 = X_train_NN.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'], axis=1, inplace=False)
df3 = X_test_NN.set_axis(['age','educationno','capitalgain','capitalloss','hoursperweek'], axis=1, inplace=False)   


# X and Y variables
X_train_N = pd.concat([X_train_var,df2],axis=1)
X_train.shape
X_test_N = pd.concat([X_test_var,df3],axis=1)
Y_train_new = Y_train_N["Salary"]
Y_test_new = Y_test_N["Salary"]

#normality
#Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(X_train_N)
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal

# Model Fitting
# 1. Bernoulli Naive Bayes 
from sklearn.naive_bayes import BernoulliNB
B=BernoulliNB()
model = B.fit(X_train_N,Y_train_new)
Y_pred=B.predict(X_test_N)

# accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm1=confusion_matrix(Y_test_new,Y_pred)
print(cm1)
acc1=accuracy_score(Y_test_new,Y_pred)
print(acc1)
# the accuracy score is 72.89%

# 2. Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
model = MNB.fit(X_train_N,Y_train_new)
Y_pred1 = MNB.predict(X_test_N)

#Metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(Y_test_new,Y_pred1)
print(cm)
acc=accuracy_score(Y_test_new,Y_pred1)
print(acc)
# the accuracy score is 76.89%
"""
 since the data is normal so we can use Gaussian Naive Bayes
 """
# 3. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB().fit(X_train_N,Y_train_new)
Y_pred2 = GNB.predict(X_test_N)

#Metrics
cm2=confusion_matrix(Y_test_new,Y_pred2)
print(cm2)
acc2=accuracy_score(Y_test_new,Y_pred2)
print(acc2)
# the accuracy score is 79.68%

"""
Inference: Hence the best fit model is with Gaussian Naive Bayes.
           Accuracy score is 79.68%
          
"""


































