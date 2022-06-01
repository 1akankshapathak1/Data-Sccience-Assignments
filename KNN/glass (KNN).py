# Prepare a model for glass classification using KNN

import pandas as pd
import numpy as np

df = pd.read_csv("glass.csv")
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

sns.countplot(x="Type", data=df)

sns.heatmap(df.isnull(), yticklabels=False, cbar=True)

# Split X and Y
X = df.iloc[:,:9]
Y = df.iloc[:,9:]
X.ndim
Y.ndim

#Standardising the data
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_scale = std.fit_transform(X)
print(X_scale)
print(Y)

#Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.30,random_state=80)

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_r_acc = []
for i in range(1,40,1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    test_score = knn.score(X_test,Y_test)
    train_score = knn.score(X_train,Y_train)
    knn_r_acc.append((i, test_score ,train_score))
df1 = pd.DataFrame(knn_r_acc, columns=['K','Test Score','Train Score'])
print(df1)
# best result at K = 7
knn=KNeighborsClassifier(n_neighbors=7,p=2)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)

# Compute confusion matrix
from sklearn import metrics
cm = metrics.confusion_matrix(Y_test,Y_pred)
print(cm)


#Metrics
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,Y_pred)

print("Accuracy of KNN with k= 7 is:",(acc*100).round(3))

# Accuracy of KNN with k= 7 is: 60.000
























