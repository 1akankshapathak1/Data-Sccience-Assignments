"""
Test of Hypothesis
"""
import pandas as pd
import numpy as np
df = pd.read_csv("E:\\assignments\\Test of Hypothesis\\BuyerRatio.csv")
df.head
df.shape
type(df)
list(df)
df.ndim
df.info()
df.isnull().sum()
df.describe()

#Hypothesis testing
'''
H0 == The male-female buyer rations are similar across regions
H1 == The male-female buyer rations are not similar across regions
'''
# Since here there is 2 rows and 5 columns with continuous and categorical data
# so we can go with propotion_test , anova_test, or chi_square_test
#we will go with Anova test

#Anova-test
import scipy.stats as stats
stat, p = stats.f_oneway(df['East'],df['West'],df['North'],df['South'])
print(stat)
print(p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

#H0 is Accepted hence male and female buyer ratio are same

