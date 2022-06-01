"""
a prediction model for profit of 50_startups data.

"""
import pandas as pd
import numpy as np

df = pd.read_csv("50_Startups.csv")
df
type(df)
df.shape
df.ndim
df.head()
df.tail()
list(df)
df.info()
df.isnull()
df.describe()
df.hist()
df.corr()

import seaborn as sns
sns.pairplot(df)

df['State'].value_counts()
y = df['Profit']

#x = df["R&D Spend"]
#x = x[:,np.newaxis]
#x.ndim

#x = df[["R&D Spend","Administration"]]

x = df[["R&D Spend","Marketing Spend","Administration"]]




# Transformation
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder()
dummies = pd.get_dummies(df.State)
df2 = pd.concat([df,pd.DataFrame(dummies)],axis=1)
df2

# Model fitting
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(x,y)
model.intercept_
model.coef_

y_pred = model.predict(x)

import numpy as np
import matplotlib.pyplot as plt
plt.scatter(y,y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');
sns.regplot(x=y,y=y_pred,ci=None,color='blue');


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
print("Mean square error",(mse).round(3))

rmse = np.sqrt(mse)
rmse

from sklearn.metrics import r2_score
r2 = r2_score(y,y_pred)*100
print("R square:", r2.round(3))

import statsmodels.api as sma
x1 = sma.add_constant(x)
lm2 = sma.OLS(y,x1).fit()
lm2.summary() 


import statsmodels.api as sm
X1 = sm.add_constant(x) ## let's add an intercept (beta_0) to our model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X1.values, j) for j in range(X1.shape[1])]
variable_VIF = pd.concat([pd.DataFrame(X1.columns),pd.DataFrame(np.transpose(vif))], axis = 1)
print(variable_VIF)

