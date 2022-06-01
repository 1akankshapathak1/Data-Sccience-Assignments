import pandas as pd
df = pd.read_csv("bank-full.csv",sep=';')
df
df.head()
list(df)
df.shape
#------------------------------------------------------------------------------
# EDA
type(df)
df.ndim
df.head()
df.tail()
df.info()
df.isnull()
df.describe()
df.corr()

import seaborn as sns
sns.pairplot(df)

df.plot.scatter(x="age", y="duration")

sns.countplot(x="y", data=df)


#-----------------------------------------------------------------------------
# Data Preprocessing

x1 = ['age', 'day', 'balance', 'duration', 'campaign', 'previous', 'pdays']
x2 =['contact','poutcome', 'job', 'marital','housing', 'loan','y']

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
df[x1] = minmax.fit_transform(df[x1])
df2 = pd.DataFrame(df[x1])
df2

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[x2] = df[x2].apply(le.fit_transform)
df3 = pd.DataFrame(df[x2])
df3

df4 = pd.concat([df2,df3],axis=1)
df4
#-----------------------------------------------------------------------------
# Data Partition

X = df4.iloc[:,:13]
Y = df4.iloc[:,13]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y, test_size=0.20,random_state=1)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
#------------------------------------------------------------------------------
# Model Fitting

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,Y)
lr.intercept_
lr.coef_

Y_Pred = lr.predict(X)


lr.fit(X_train,Y_train)
Y_pred_train = lr.predict(X_train)
Y_pred_test  = lr.predict(X_test)


import matplotlib.pyplot as plt
plt.scatter(Y,Y_Pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');
sns.regplot(x=Y,y=Y_Pred,ci=None,color='red');

#-----------------------------------------------------------------------------
# Sklearn Calculations

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

CM = confusion_matrix(Y,Y_Pred)
CM

TN = CM[0,0]
FN = CM[1,0]
FP = CM[0,1]
TP = CM[1,1]

print("Accuracy score:",(accuracy_score(Y,Y_Pred)*100).round(3)) 
# accuracy score = 89.038

print("Recall/Senstivity score:",(recall_score(Y,Y_Pred)*100).round(3))
# recall score = 18.926

print("Precision score:",(precision_score(Y,Y_Pred)*100).round(3))
# precision score = 59.976

Specificity = TN / (TN + FP)
print("Specificity score:",(Specificity*100).round(3))
# specificity score = 98.327

print("F1 score:",(f1_score(Y,Y_Pred)*100).round(3))
# f1 score = 28.773
 
#-----------------------------------------------------------------------------
# Training and Test Accuracies

Training_accuracy = accuracy_score(Y_train,Y_pred_train)
Test_accuracy = accuracy_score(Y_test,Y_pred_test)
print("Accuracy Score on Training data: ",Training_accuracy.round(3))
# training accuracy score = 89.1

print("Accuracy Score on Test data: ",Test_accuracy.round(3))
# test accuracy score = 89
#------------------------------------------------------------------------------
# Cross Validation

from sklearn.model_selection import cross_val_score
lr = LogisticRegression()
results = cross_val_score(lr, X, Y,scoring='accuracy', cv=100)
abs(results)

import numpy as np
np.mean(abs(results))
# accuracy (cross validation) = 88.978
#-----------------------------------------------------------------------------
# confusion matrix in matplotlib

import matplotlib.pyplot as plt 
plt.matshow(CM)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#-----------------------------------------------------------------------------
# ROC Curve

from sklearn.metrics import roc_curve,roc_auc_score
lr.predict_proba(X).shape

lr.predict_proba(X)[:,1]

print(lr.predict_proba(X))
y_pred_proba = lr.predict_proba(X)[:,1]
FPR, TPR,_ = roc_curve(Y,y_pred_proba)

plt.plot(FPR,TPR)

plt.ylabel('TPR - True Positive Rate')
plt.xlabel('FPR - False Positive Rate')
#----------------------------------------------------------------------------
# Summary

import statsmodels.api as sma
log_reg = sma.Logit(Y,X).fit()
print(log_reg.summary())

#============================================================================






