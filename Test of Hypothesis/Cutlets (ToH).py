"""
A F&B manager wants to determine whether there is any significant difference 
in the diameter of the cutlet between two units. A randomly selected sample
of cutlets was collected from both units and measured? Analyze the 
data and draw inferences at 5% significance level. 
Please state the assumptions and tests that you carried out to 
check validity of the assumptions.
"""
import pandas as pd
import numpy as np
df = pd.read_csv("E:\\assignments\\Test of Hypothesis\\Cutlets.csv")
df.shape
df.head
list(df)
type(df)
df.ndim
df.info()
df.describe()
df.isnull().sum()

# Visaulisation
import seaborn as sns
sns.distplot(df['Unit A'])
df['Unit A'].hist() #Here as per the histogram, it looks like positively skewed
df['Unit A'].skew() #Skewness is -0.012, it can be accpeted as it is under range of -0.5 to +0.5


sns.distplot(df['Unit B'])
df['Unit B'].hist()   #Here as per the histogram, it looks like positively skewed
df['Unit B'].skew()   #Skewness is -0.37, it can be accpeted as it is under range of -0.5 to +0.5


sns.pairplot(df)

## Box and Whisker Plots
import matplotlib.pyplot as plt
plt.boxplot(df['Unit A'])
plt.boxplot(df['Unit B'])

  
# scatter plot
df.plot.scatter(x='Unit A', y='Unit B')

df.corr()

'''
# test of hypothesis  --> Normality test
Ho: Data is normal
H1: Data is not normal
alpha = 0.5  
'''
# shapiro test
from scipy.stats import shapiro
stat_A, P_a = shapiro(df['Unit A'])
stat_B, P_b = shapiro(df["Unit B"])
print(stat_A,stat_B)
print("p-value",P_a,P_b)

alpha = 0.05 # 5% level of significance

if P_a < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
if P_b < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
# H0: Data is normal 

'''now we have check is the mean of Unit A and Unit B are same 
H0 == Mean for Unit A and Unit B are equal (There is no significance difference between diameter of the Culets)
H1 == Mean for Unit A and Unit B are not equal (There is a significance difference between diameter of the Culets)
'''# here we need to do two propotion test 

df.mean()  #mean of Unit A and Unit B

from scipy import stats
#from statsmodels.stats import weightstats as stats
ztest ,pval = stats.ttest_ind(df['Unit A'],df['Unit B'])
print("Zcalcualted value is ",ztest.round(3))
print("P-value value is ",pval.round(3))

alpha = 0.05 # 5% level of significance

if pval < alpha:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and H0 is accepted")
    
#H0 is accepted.hence Mean of both Unit A and Unit B are equal
