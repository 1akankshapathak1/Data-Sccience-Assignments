# Implement a KNN model to classify the animals in to categorie

import pandas as pd
import numpy as np
df = pd.read_csv("Zoo.csv")
df

# EDA
type(df)
df.shape
df.ndim
df.head()
df.tail()
list(df)
df.info()
df.isnull()
df.describe()
df.corr()

import seaborn as sns
sns.pairplot(df)

sns.countplot(x="type", data=df)

# Data wrangling
df2 = df.drop('animal name',axis=1)
df2

# Data splitting
Y = df['type']
X = df.iloc[:,1:17]

pd.crosstab(Y,Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,stratify=Y,test_size=0.20, random_state=25)

# Model Fitting
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Calculating accuracies for train and test for different k values
knn_r_acc = []
for i in range(1,40,1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    test_score = knn.score(X_test,Y_test)
    train_score = knn.score(X_train,Y_train)
    knn_r_acc.append((i, test_score ,train_score))
df1 = pd.DataFrame(knn_r_acc, columns=['K','Test Score','Train Score'])
print(df1)

# best result at K = 5
knn = KNeighborsClassifier(n_neighbors=5,p=1)
# K = 5, p=2 is eucledian distance by default
knn.fit(X_train,Y_train)
# prediction
y_pred=knn.predict(X_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(Y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred).round(3)
# accuracy score = 92.3%



















