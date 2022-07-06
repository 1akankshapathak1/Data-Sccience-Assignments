"""
Forecast the Airlines Passengers data set. Prepare a document for each model
explaining how many dummy variables you have created and RMSE value for each
model.Finally which model you will use for Forecasting. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel('E:\\assignments\\Forecasting\\Airlines+Data.xlsx')
data.shape
list(data)
data.head

t=[]
for i in range(1,97,1):
    t.append(i)

data['t']=pd.DataFrame(t)
data['t_square']=np.square(data['t'])
data['log_pass']=np.log(data['Passengers'])

#Extracting months
data['month']=data['Month'].dt.strftime('%b')

#Extracting years
data['year']=data['Month'].dt.strftime('%Y')

#Getting dummies of month column
data=pd.get_dummies(data,columns=['month'])
data['month']=data['Month'].dt.strftime('%b')
list(data)

#Plots
#Heatmap
%matplotlib qt
plt.figure(figsize=(12,8))
heatmap_month=pd.pivot_table(data=data,values='Passengers',index='year',columns='month',aggfunc='mean',fill_value=0)
sns.heatmap(heatmap_month,annot=True,fmt='g')

#Boxplot
%matplotlib qt
plt.figure(figsize=(12,8))
sns.boxplot(x='month',y='Passengers',data=data)
sns.boxplot(x='year',y='Passengers',data=data)

#Line plot
%matplotlib qt
plt.figure(figsize=(12,8))
sns.lineplot(x='month',y='Passengers',data=data)
sns.lineplot(x='year',y='Passengers',data=data)

#Splitting the data into train and test
data.shape
list(data)
train=data.head(70)
list(train)
test=data.tail(26)
list(test)

#Fitting the model 
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error

#1) Linear model
lin_model=smf.ols('Passengers~t',data=train).fit()
lin_pred=pd.Series(lin_model.predict(test['t']))
lin_MSE=mean_squared_error(test['Passengers'],lin_pred)
lin_RMSE=np.sqrt(lin_MSE)
print(lin_RMSE)

#2) Exponential model
exp_model=smf.ols('log_pass~t',data=train).fit()
exp_pred=pd.Series(exp_model.predict(test['t']))
exp_MSE=mean_squared_error(test['Passengers'],exp_pred)
exp_RMSE=np.sqrt(exp_MSE)
print(exp_RMSE)

#3) Quadratic model
quad_model=smf.ols('Passengers~t+t_square',data=train).fit()
quad_pred=pd.Series(quad_model.predict(test[['t','t_square']]))
quad_MSE=mean_squared_error(test['Passengers'],quad_pred)
quad_RMSE=np.sqrt(quad_MSE)
print(quad_RMSE)

#4) Additivie seasonality model
add_seas_model=smf.ols('Passengers~month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=train).fit()
add_seas_pred=pd.Series(add_seas_model.predict(test[['month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
add_seas_MSE=mean_squared_error(test['Passengers'],add_seas_pred)
add_seas_RMSE=np.sqrt(add_seas_MSE)
print(add_seas_RMSE)

#5) Additive seasonality Quadratic model
add_seas_quad_model=smf.ols('Passengers~t+t_square+month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=train).fit()
add_seas_quad_pred=pd.Series(add_seas_quad_model.predict(test[['t','t_square','month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
add_seas_quad_MSE=mean_squared_error(test['Passengers'],add_seas_quad_pred)
add_seas_quad_RMSE=np.sqrt(add_seas_quad_MSE)
print(add_seas_quad_RMSE)

#6) Multiplicative seasonality model
mul_seas_model=smf.ols('log_pass~month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=train).fit()
mul_seas_pred=pd.Series(mul_seas_model.predict(test[['month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
mul_seas_MSE=mean_squared_error(test['Passengers'],mul_seas_pred)
mul_seas_RMSE=np.sqrt(mul_seas_MSE)
print(mul_seas_RMSE)

#7) Multiplicative Additive seasonality model
mul_add_seas_model=smf.ols('log_pass~t+month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=train).fit()
mul_add_seas_pred=pd.Series(mul_add_seas_model.predict(test[['t','month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
mul_add_seas_MSE=mean_squared_error(test['Passengers'],mul_add_seas_pred)
mul_add_seas_RMSE=np.sqrt(mul_add_seas_MSE)
print(mul_add_seas_RMSE)

#8) Multiplicative Additive Quadratic seasonality model
mul_add_quad_seas_model=smf.ols('log_pass~t+t_square+month_Apr+month_Aug+month_Dec+month_Feb+month_Jan+month_Jul+month_Jun+month_Mar+month_May+month_Nov+month_Oct+month_Sep',data=train).fit()
mul_add_quad_seas_pred=pd.Series(mul_add_quad_seas_model.predict(test[['t','t_square','month_Apr','month_Aug','month_Dec','month_Feb','month_Jan','month_Jul','month_Jun','month_Mar','month_May','month_Nov','month_Oct','month_Sep']]))
mul_add_quad_seas_MSE=mean_squared_error(test['Passengers'],mul_add_quad_seas_pred)
mul_add_quad_seas_RMSE=np.sqrt(mul_add_quad_seas_MSE)
print(mul_add_quad_seas_RMSE)

#Comparing model results
table={'model':pd.Series(['lin_model','exp_model','quad_model','add_seas_model','add_seas_quad_model','mul_seas_model','mul_add_seas_model','mul_add_quad_seas_model']),'RMSE':pd.Series([lin_RMSE,exp_RMSE,quad_RMSE,add_seas_RMSE,add_seas_quad_RMSE,mul_seas_RMSE,mul_add_seas_RMSE,mul_add_quad_seas_RMSE])}
table_new=pd.DataFrame(table)
table_new.sort_values(['RMSE'])

'''
Inference: The best fit model is Additive seasonality quadratic model with RMSE
           of 30.393 which is the least among all the models.
'''

