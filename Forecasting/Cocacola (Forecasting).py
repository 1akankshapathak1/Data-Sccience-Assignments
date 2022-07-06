"""
Forecast the CocaCola prices data set. Prepare a document for each model 
explaining how many dummy variables you have created and RMSE value for each
model. Finally which model you will use for Forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel('E:\\assignments\\Forecasting\\CocaCola_Sales_Rawdata.xlsx')
list(data)

dates=pd.date_range(start='1986',periods=42,freq='Q')
data['Dates']=pd.DataFrame(dates)
data['Sales'].plot()
data.drop(['Quarter'],axis=1,inplace=True)
data.shape
list(data)

t=[]
for i in range(1,43,1):
    t.append(i)

data['t']=pd.DataFrame(t)
data['t_square']=np.square(data['t'])
data['log_sales']=np.log(data['Sales'])

#Extracting months
data['month']=data['Dates'].dt.strftime('%b')

#Extracting years
data['year']=data['Dates'].dt.strftime('%Y')

#Getting dummies of month column
data=pd.get_dummies(data,columns=['month'])
data['month']=data['Dates'].dt.strftime('%b')
list(data)

#Plots
#Heatmap
%matplotlib qt
plt.figure(figsize=(12,8))
heatmap_month=pd.pivot_table(data=data,values='Sales',index='year',columns='month',aggfunc='mean',fill_value=0)
sns.heatmap(heatmap_month,annot=True,fmt='g')

#Boxplot
%matplotlib qt
plt.figure(figsize=(12,8))
sns.boxplot(x='month',y='Sales',data=data)
sns.boxplot(x='year',y='Sales',data=data)

#Line plot
%matplotlib qt
plt.figure(figsize=(12,8))
sns.lineplot(x='month',y='Sales',data=data)
sns.lineplot(x='year',y='Sales',data=data)

#Splitting the data into train and test
list(data)
train=data.head(32)
list(train)
test=data.tail(10)
list(test)

#Fitting the model 
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

#1) Linear model
lin_model=smf.ols('Sales~t',data=train).fit()
lin_pred=pd.Series(lin_model.predict(test['t']))
lin_MSE=mean_squared_error(test['Sales'],lin_pred)
lin_RMSE=np.sqrt(lin_MSE)
print(lin_RMSE)

#2) Exponential model
exp_model=smf.ols('log_sales~t',data=train).fit()
exp_pred=pd.Series(exp_model.predict(test['t']))
exp_MSE=mean_squared_error(test['Sales'],exp_pred)
exp_RMSE=np.sqrt(exp_MSE)
print(exp_RMSE)

#3) Quadratic model
quad_model=smf.ols('Sales~t+t_square',data=train).fit()
quad_pred=pd.Series(quad_model.predict(test[['t','t_square']]))
quad_MSE=mean_squared_error(test['Sales'],quad_pred)
quad_RMSE=np.sqrt(quad_MSE)
print(quad_RMSE)

#4) Additivie seasonality model
add_seas_model=smf.ols('Sales~month_Dec+month_Jun+month_Mar+month_Sep',data=train).fit()
add_seas_pred=pd.Series(add_seas_model.predict(test[['month_Dec','month_Jun','month_Mar','month_Sep']]))
add_seas_MSE=mean_squared_error(test['Sales'],add_seas_pred)
add_seas_RMSE=np.sqrt(add_seas_MSE)
print(add_seas_RMSE)

#5) Additive seasonality Quadratic model
add_seas_quad_model=smf.ols('Sales~t+t_square+month_Dec+month_Jun+month_Mar+month_Sep',data=train).fit()
add_seas_quad_pred=pd.Series(add_seas_quad_model.predict(test[['t','t_square','month_Dec','month_Jun','month_Mar','month_Sep']]))
add_seas_quad_MSE=mean_squared_error(test['Sales'],add_seas_quad_pred)
add_seas_quad_RMSE=np.sqrt(add_seas_quad_MSE)
print(add_seas_quad_RMSE)

#6) Multiplicative seasonality model
mul_seas_model=smf.ols('log_sales~month_Dec+month_Jun+month_Mar+month_Sep',data=train).fit()
mul_seas_pred=pd.Series(mul_seas_model.predict(test[['month_Dec','month_Jun','month_Mar','month_Sep']]))
mul_seas_MSE=mean_squared_error(test['Sales'],mul_seas_pred)
mul_seas_RMSE=np.sqrt(mul_seas_MSE)
print(mul_seas_RMSE)

#7) Multiplicative Additive seasonality model
mul_add_seas_model=smf.ols('log_sales~t+month_Dec+month_Jun+month_Mar+month_Sep',data=train).fit()
mul_add_seas_pred=pd.Series(mul_add_seas_model.predict(test[['t','month_Dec','month_Jun','month_Mar','month_Sep']]))
mul_add_seas_MSE=mean_squared_error(test['Sales'],mul_add_seas_pred)
mul_add_seas_RMSE=np.sqrt(mul_add_seas_MSE)
print(mul_add_seas_RMSE)

#8) Multiplicative Additive Quadratic seasonality model
mul_add_quad_seas_model=smf.ols('log_sales~t+t_square+month_Dec+month_Jun+month_Mar+month_Sep',data=train).fit()
mul_add_quad_seas_pred=pd.Series(mul_add_quad_seas_model.predict(test[['t','t_square','month_Dec','month_Jun','month_Mar','month_Sep']]))
mul_add_quad_seas_MSE=mean_squared_error(test['Sales'],mul_add_quad_seas_pred)
mul_add_quad_seas_RMSE=np.sqrt(mul_add_quad_seas_MSE)
print(mul_add_quad_seas_RMSE)

#Comparing model results
table={'model':pd.Series(['lin_model','exp_model','quad_model','add_seas_model','add_seas_quad_model','mul_seas_model','mul_add_seas_model','mul_add_quad_seas_model']),'RMSE':pd.Series([lin_RMSE,exp_RMSE,quad_RMSE,add_seas_RMSE,add_seas_quad_RMSE,mul_seas_RMSE,mul_add_seas_RMSE,mul_add_quad_seas_RMSE])}
table_new=pd.DataFrame(table)
table_new.sort_values(['RMSE'])

'''
Inference: The best fit model is Additive seasonality quadratic model with RMSE
           of 277.35 which is the least among all the models.
'''

