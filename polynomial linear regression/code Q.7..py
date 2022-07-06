# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:04:58 2022

@author: hp
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("C:\\Users\\hp\\Downloads\\Round 2 Assignment\\Round 2 Assignment\\Q7\\data\\modelling_data.xlsx")

type(df)
df.shape
df.ndim
df.head()
df.tail()
list(df)
df.info()
df.isnull().sum()
df.describe()
df.corr()

X = df.iloc[:,1]
X = X[:, np.newaxis]
X.ndim

Y = df.iloc[:, 0]

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,Y)

# Visualising the Linear Regression results
plt.scatter(X, Y, color = 'blue')

plt.plot(X, lr.predict(X), color = 'red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(X)

poly.fit(X_poly, Y)
lr2 = LinearRegression()
model = lr2.fit(X_poly, Y)
y_pred = model.predict(X_poly)
print(y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,y_pred)
print("Mean square error:",mse.round(3))

RMSE = np.sqrt(mse)
print("RMSE:",RMSE.round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,y_pred)*100
print("R square:", r2.round(3))

# Visualising the Polynomial Regression results
plt.scatter(X, Y, color = 'blue')

plt.plot(X, lr2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()


"""
Inference:
    For degree i = 48 we can have the highest R- Squared.
    But we avoid taking i>3 because this results in overfitting of the model
    and unreliable results.
"""
    















