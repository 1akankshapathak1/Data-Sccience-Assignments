"""
multiple linear regression
prediction model for predicting Price.

"""

import pandas as pd
df = pd.read_csv("ToyotaCorolla.csv")
df.drop(['Id','Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee',
        'BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer',
        'CD_Player','Central_Lock','Powered_Windows', 'Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider',
         'Metallic_Rim', 'Radio_cassette', 'Tow_Bar'],axis = 1,inplace = True)
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
df.corr()

import seaborn as sns
sns.pairplot(df)

# Test of hypothesis 
from scipy.stats import shapiro
stat, p = shapiro(df['Price'])
print(stat)
print("p-value",p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

# H1: Data is normal


# Splitting of Data
Y = df['Price']
x = df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]


# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scale = scaler.fit_transform(x)
x_new = pd.DataFrame(X_scale)
X1 =  x_new.set_axis([["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]],axis=1)


#X = df[['Age_08_04','KM','HP','Gears','Quarterly_Tax','Weight']]
#X = df[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']]  #P val Doors = 0.968>0.005
X = df[['Age_08_04','KM','HP','cc','Gears','Quarterly_Tax','Weight']] #P val cc = 0.169>0.005 
#X = df[['Age_08_04','KM','HP','cc','Quarterly_Tax','Weight']] #P val 
#X = df[['Age_08_04','KM','HP','Quarterly_Tax','Weight']]
#X = df[['Age_08_04','KM','HP','Gears','Quarterly_Tax','Weight']], PERFECTLY FIT





from sklearn.linear_model import LinearRegression
LM = LinearRegression()
LM.fit(X,Y)
Y_pred  = LM.predict(X)
LM.intercept_
LM.coef_


import numpy as np

import matplotlib.pyplot as plt
plt.scatter(Y,Y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');
sns.regplot(x=Y,y=Y_pred,ci=None,color='red');

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)

print("Mean square error: ", (mse).round(3))

from sklearn.metrics import r2_score
r2 = r2_score(Y,Y_pred)*100
print("R square: ", r2.round(3))

import statsmodels.api as sma
x1 = sma.add_constant(X)
lm2 = sma.OLS(Y,x1).fit()
lm2.summary()


import statsmodels.api as sm
X1 = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X1.values, j) for j in range(X1.shape[1])]
variable_VIF = pd.concat([pd.DataFrame(X1.columns),pd.DataFrame(np.transpose(vif))], axis = 1)
print(variable_VIF)











