"""
A hospital wants to determine whether there is any difference in the average
Turn Around Time (TAT) of reports of the laboratories on their preferred list.
They collected a random sample and recorded TAT for reports of 4 laboratories. 
TAT is defined as sample collected to report dispatch.
Analyze the data and determine whether there is any difference in average 
TAT among the different laboratories at 5% significance level.
"""
import pandas as pd
import numpy as np
df = pd.read_csv("E:\assignments\Test of Hypothesis\\LabTAT.csv")
df.head
df.shape
type(df)
list(df)
df.ndim
df.info()
df.isnull().sum()
df.describe()
df.hist()
df.skew()

# Data Visualisation
#Pairplot
import seaborn as sns
sns.pairplot(df)

#displot
sns.displot(df)

#Box and Whisker Plots
import matplotlib.pyplot as plt
plt.boxplot(df)


#checking the co relation
df.corr()

'''
# test of hypothesis  --> Normality test
Ho: Data is normal
H1: Data is not normal
alpha = 0.5  
'''
# shapiro test
from scipy.stats import shapiro
stat_1, P_1 = shapiro(df['Laboratory 1'])
stat_2, P_2 = shapiro(df["Laboratory 2"])
stat_3, P_3 = shapiro(df["Laboratory 3"])
stat_4, P_4 = shapiro(df["Laboratory 4"])
print(stat_1,stat_2,stat_3,stat_4)
print("p-value",P_1,P_2,P_3,P_4)

alpha = 0.05 # 5% level of significance

if P_1 < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
if P_2 < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

if P_3 < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")

if P_4 < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")    

# H0:is Accepted hence Data is normal 

'''now we have check is the mean of Laboratory 1,Laboratory 2,Laboratory 3 and Laboratory 4 are same 
H0 == Mean of Laboratory 1,Laboratory 2,Laboratory 3 and Laboratory 4 are equal (There is no significance difference between TAT)
H1 == Mean of any one Laboratory 1,Laboratory 2,Laboratory 3,and Laboratory 4  are not equal (There is a significance difference between TAT)
'''# here we need to do Anova test 

import scipy.stats as stats
stat, p = stats.f_oneway(df['Laboratory 1'],df['Laboratory 2'],df['Laboratory 3'],df['Laboratory 4'])
print(stat)
print(p)

alpha = 0.05 # 5% level of significance

if p < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
# H1: is Accepted hence TAT of Laboratory are not Equal


