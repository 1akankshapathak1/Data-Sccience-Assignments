
"""
Use Random Forest to prepare a model on fraud data 
treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("E:\\assignments\\Random Forest\\Fraud_check (1).csv")
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
df.isnull().sum()
df.describe()
df.corr()

# Data Visualisation

#Countplot for categorical variables
sns.countplot(df['Undergrad'])
sns.countplot(df['Marital.Status'])
sns.countplot(df['Urban'])

#Histograms for numerical variables
sns.distplot(df['Taxable.Income'])
sns.distplot(df['City.Population'])
sns.distplot(df['Work.Experience'])

# Splitting of Data
# X Variables
X1 = df[['Undergrad', 'Marital.Status', 'Urban']]
X2 = df[['City.Population', 'Work.Experience']]

# Y variable.
Y=df['Taxable.Income']
df.drop(['Taxable.Income'],axis=1,inplace=True)
df.describe()

#Converting Y variable into categorical 
Y1=[]
for i in range(0,600,1):
    if Y.iloc[i,]<=30000:
        print('Risky')
        Y1.append('Risky')
    else:
        print('Good')
        Y1.append('Good')


Y_new=pd.DataFrame(Y1)
Y_new[0].ndim
list(Y_new)
Y_new.set_axis(['Taxable.Income'],axis='columns',inplace=True)
sns.countplot(Y_new['Taxable.Income'])

Y_new.ndim
Y_new['Taxable.Income'].ndim

# Data Transformation
from sklearn.preprocessing import LabelEncoder, StandardScaler
LE = LabelEncoder()
SS = StandardScaler()

for i in range(0,3,1):
    X1.iloc[:,i] = LE.fit_transform(X1.iloc[:,i])
    X1.iloc[:,i] = LE.fir_transform(X1.iloc[:,i])

X2 = SS.fit_transform(X2)
X2_n = pd.DataFrame(X2)
X2_n.set_axis(['City.Population','Work.Experience'], axis=1, inplace=True)

X = pd.concat([X1,X2_n], axis=1)

#Splitting into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y_new['Taxable.Income'],stratify=Y_new['Taxable.Income'],test_size=0.25,random_state=67)
X_train.shape
Y_train.shape
Y_new.ndim

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(max_features=0.30000000000000004,n_estimators=500)
model=RFC.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

#Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error
acc=accuracy_score(Y_test, Y_pred)
print(acc)

acc1=[]
set1=np.arange(0.1,1.1,0.1)
for i in set1:
    RFC=RandomForestClassifier(max_features=i,n_estimators=500)
    model = RFC.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    acc=accuracy_score(Y_test, Y_pred)
    acc1.append((acc*100).round(3))
    print('For max features',i,',accuracy is',(acc*100).round(3))


plt.plot(set1,acc1,data=None)
plt.xlabel('max_features')
plt.ylabel('accuracy')
plt.title('Graph between max features and accuracy')
plt.show()

"""
Inference: Accuracy for max_features 0.30000000000000004 is 79.333 for 
           randon state 67
"""






