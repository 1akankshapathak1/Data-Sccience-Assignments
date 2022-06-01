"""
  Salary_hike -> Build a prediction model for Salary_hike


"""

import pandas as pd
import numpy as np

df = pd.read_csv("Salary_Data.csv")
df
df.shape
type(df)
list(df)
df.ndim
df.info()
df.head()
df.tail()
df.describe()
df.hist()
df.plot.scatter(x='Salary',y='YearsExperience')
df.boxplot()
df.corr()

# shapiro test

from scipy.stats import shapiro
stat, p = shapiro(df['Salary'])
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal

x = df['Salary']
x.ndim
x = x[:, np.newaxis]
x.ndim

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(x)

X = pd.DataFrame(X_scale)



y = df['YearsExperience']
y.shape
y.ndim




from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)
model.intercept_
model.coef_

y_pred = model.predict(X)
print(y_pred)

import matplotlib.pyplot as plt
plt.scatter(X,y, color = 'black')
plt.plot(X_scale,y_pred, color = 'green')
plt.show()

y_error = y - y_pred
print(y_error)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
print("Mean square error:",mse.round(3))

RMSE = np.sqrt(mse)
print("RMSE:",RMSE.round(3))

from sklearn.metrics import r2_score
r2 = r2_score(y,y_pred)*100
print("R square:", r2.round(3))

import statsmodels.api as sma
x1 = sma.add_constant(X)
lm2 = sma.OLS(y,x1).fit()
lm2.summary()























