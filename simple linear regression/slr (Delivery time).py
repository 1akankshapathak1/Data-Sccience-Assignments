"""
 Delivery_time -> Predict delivery time using sorting time 

"""

import pandas as pd
import numpy as np

df = pd.read_csv("delivery_time.csv")
df
df.shape
type(df)
list(df)
df.ndim
df.hist()
df.describe()

#X = df['Sorting Time'].transform(func = lambda x : x**2)
#X = df['Sorting Time'].transform(func = lambda x : x//2)
#X = df['Sorting Time'].transform(func = lambda x : np.sqrt(x))
X = df['Sorting Time'].transform(func = lambda x : np.log(x))
print(X)

df['Delivery Time'].mean()

'''
Ho:   mu =  X bar
H1:   mu != X bar
'''    

from statsmodels.stats import weightstats as stests
zcalc ,pval = stests.ztest(df['Delivery Time'], x2=None, value=8,alternative='two-sided')
print("Zcalcualted value is ",zcalc.round(4))
print("P-value value is ",pval.round(4))

alpha = 0.05 # 5% level of significance

if pval < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

y = df["Delivery Time"]
#x = df["Sorting Time"]
X = X[:, np.newaxis]
X.ndim



df.plot.scatter(x='Sorting Time',y='Delivery Time')
df.corr()
                     
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X,y)
model.intercept_
model.coef_

y_pred = model.predict(X)
print(y_pred)

import matplotlib.pyplot as plt
plt.scatter(X,y, color = 'green')
plt.plot(X,y_pred, color = 'blue')
plt.show()

y_error = y - y_pred
print(y_error)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
print("Mean square error:",mse.round(3))

RMSE = np.sqrt(mse)
print("Root mean square:",RMSE.round(3))

from sklearn.metrics import r2_score
r2 = r2_score(y,y_pred)*100
print("R square:", r2.round(3))

import statsmodels.api as sma
x1 = sma.add_constant(X)
lm2 = sma.OLS(y,x1).fit()
lm2.summary()

