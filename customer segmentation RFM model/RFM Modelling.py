"""
Customer Segmentation using RFM model
- created by Akanksha Pathak
"""
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns 
import matplotlib.pyplot as plt

data = pd.read_excel("C:\\Users\\hp\\Downloads\\Round 1 Assignment\\sales_data.xlsx")

# EDA
data.shape
data.ndim
type(data)
data.head()
list(data)
data.info()

# Checking the missing values
null_values = pd.DataFrame(data.isnull().sum(),columns=['count_value']) 

data.describe()
data.corr()

%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)

# Checking distribution of Total Orders
sns.violinplot(data.TOTAL_ORDERS)
data.TOTAL_ORDERS.describe()

# Checking distribution of Revenue
sns.violinplot(data.REVENUE)
data.REVENUE.describe()

# Order and revenue
data.plot.scatter(x='TOTAL_ORDERS',y='REVENUE')
#we can observ that the more cousomer has did order in range of 1 to 60 and the revenue also in between 0 to 10000

# Total  order
pd.crosstab(data.CustomerID,data.TOTAL_ORDERS).plot(kind = 'line')
"we have coustomer who has did 156 order as well "

# Average order value
sns.countplot(data['AVERAGE_ORDER_VALUE']) 
sns.countplot(data['CARRIAGE_REVENUE'])
sns.countplot(data['AVERAGESHIPPING'])

# Dayvise order and revenue observation
sns.countplot(data['MONDAY_ORDERS'])
sns.distplot(data['MONDAY_REVENUE'])
sns.countplot(data['TUESDAY_ORDERS'])
sns.distplot(data['TUESDAY_REVENUE'])
sns.countplot(data['WEDNESDAY_ORDERS'])
sns.distplot(data['WEDNESDAY_REVENUE'])
sns.countplot(data['THURSDAY_ORDERS'])
sns.distplot(data['THURSDAY_REVENUE'])
sns.countplot(data['FRIDAY_ORDERS'])
sns.distplot(data['FRIDAY_REVENUE'])
sns.countplot(data['SATURDAY_ORDERS'])
sns.distplot(data['SATURDAY_REVENUE'])
sns.countplot(data['SUNDAY_ORDERS'])
sns.distplot(data['SUNDAY_REVENUE'])

"""
so from the above graph we can understand Thursday, Saturday, Sunday coustomer 
have ordered more than other days
and on an average cost of 10000 customers order per day.
"""

# Weekwise order and revenue observation
sns.countplot(data['WEEK1_DAY01_DAY07_ORDERS'])
sns.countplot(data['WEEK2_DAY08_DAY15_ORDERS'])
sns.countplot(data['WEEK3_DAY16_DAY23_ORDERS'])
sns.countplot(data['WEEK4_DAY24_DAY31_ORDERS'])

sns.distplot(data['WEEK1_DAY01_DAY07_REVENUE'])
sns.distplot(data['WEEK2_DAY08_DAY15_REVENUE'])
sns.distplot(data['WEEK3_DAY16_DAY23_REVENUE'])
sns.distplot(data['WEEK4_DAY24_DAY31_REVENUE'])

"""
Week wise order visualisation tells us that the most order is done on 
week 2 and week 4.
It is clear from the week wise revenue visualisation that in a week 
each customer order between the cost 0-2000
"""

# Timewise order and revenue observation
sns.countplot(data['TIME_0000_0600_ORDERS'])
sns.countplot(data['TIME_0601_1200_ORDERS'])
sns.countplot(data['TIME_1200_1800_ORDERS'])
sns.countplot(data['TIME_1801_2359_ORDERS'])

sns.distplot(data['TIME_0000_0600_REVENUE'])
sns.distplot(data['TIME_0601_1200_REVENUE'])
sns.distplot(data['TIME_1200_1800_REVENUE'])
sns.distplot(data['TIME_1801_2359_REVENUE'])

"""
From time wise order visualisation we come know that the maximum order is
placed between 18:01 – 23:59.
Time wise revenue visualisation shows that the customers have ordered of
cost around 0-2000 mostly between the time period of 18:01 – 23:59.

"""

# Splitting Data
Recency_data = data[['CustomerID', 'FIRST_ORDER_DATE', 'LATEST_ORDER_DATE']]


pd.crosstab(data.MONDAY_ORDERS,data.TUESDAY_ORDERS).plot(kind='bar')('MONDAY_ORDERS','TUESDAY_ORDERS','WEDNESDAY_ORDERS','THURSDAY_ORDERS','FRIDAY_ORDERS','SATURDAY_ORDERS','SUNDAY_ORDERS')

Frequency_data = data[['CustomerID','TOTAL_ORDERS','AVERAGE_ORDER_VALUE','MONDAY_ORDERS','TUESDAY_ORDERS','WEDNESDAY_ORDERS',
                      'THURSDAY_ORDERS','FRIDAY_ORDERS','SATURDAY_ORDERS','SUNDAY_ORDERS',
            'WEEK1_DAY01_DAY07_ORDERS','WEEK2_DAY08_DAY15_ORDERS','WEEK3_DAY16_DAY23_ORDERS',
           'WEEK4_DAY24_DAY31_ORDERS','TIME_0000_0600_ORDERS','TIME_0601_1200_ORDERS',
           'TIME_1200_1800_ORDERS','TIME_1801_2359_ORDERS', ]]

Monetary_data = data[['CustomerID','REVENUE','MONDAY_REVENUE','TUESDAY_REVENUE','WEDNESDAY_REVENUE',
              'THURSDAY_REVENUE','FRIDAY_REVENUE','SATURDAY_REVENUE','SUNDAY_REVENUE',
            'WEEK1_DAY01_DAY07_REVENUE','WEEK2_DAY08_DAY15_REVENUE','WEEK3_DAY16_DAY23_REVENUE',
            'WEEK4_DAY24_DAY31_REVENUE','TIME_0000_0600_REVENUE','TIME_0601_1200_REVENUE',
            'TIME_1200_1800_REVENUE','TIME_1801_2359_REVENUE']]



#--------------------------------------------------------------------------------------------
# Recency Calculation
recency_df = Recency_data.groupby(by='CustomerID',
                        as_index=False)['LATEST_ORDER_DATE'].max()
recency_df.columns = ['CustomerID', 'LastPurchaseDate']
recent_date = recency_df['LastPurchaseDate'].max()
recency_df['Recency'] = recency_df['LastPurchaseDate'].apply(
    lambda x: (recent_date - x).days)
recency_df.head()

# Recency and customer id
sns.barplot(x='CustomerID',y='Recency',data = recency_df,color='red',)
recency_df.plot.line(x='CustomerID',y='Recency')

#---------------------------------------------------------------------------------------------
# Frequency Calculation
frequency_df = Frequency_data.drop_duplicates(subset=['TOTAL_ORDERS', 'CustomerID'],
           keep="first", inplace=True)
frequency_df = Frequency_data.groupby(by=['CustomerID'], as_index=False)['TOTAL_ORDERS'].count()
frequency_df.columns = ['CustomerID', 'Frequency']
frequency_df.head()
frequency_df['Frequency'].unique()

# Plotting frequency and customer id
sns.barplot(x='CustomerID',y='Frequency',data = frequency_df,color='red',)
frequency_df.plot.line(x='CustomerID',y='Frequency')
#--------------------------------------------------------------------------------------------
# Monetary Calculation

monetary_df = Monetary_data.groupby(by='CustomerID', as_index=False)['REVENUE'].sum()
monetary_df.columns = ['CustomerID', 'Monetary']
monetary_df.head()

# Plotting Monetary and customer id
sns.barplot(x='CustomerID',y='Monetary',data = monetary_df,color='red',)
monetary_df.plot.line(x='CustomerID',y='Monetary')
#-----------------------------------------------------------------------

#---------------------------------------------------------------------------

# RFM Table
temp_df = recency_df.merge(frequency_df,on='CustomerID')
temp_df.head()
#merge with monetary dataframe to get a table with the 3 columns
rfm_df = temp_df.merge(monetary_df,on='CustomerID').drop(columns='LastPurchaseDate')
rfm_df.head()

# Ranking Customer’s based upon their recency, frequency, and monetary score
rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)

# normalizing the rank of the customers
rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100

rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)

rfm_df.head()

# Calculating RFM Score
rfm_df['RFM_Score'] = 0.15*rfm_df['R_rank_norm']+0.28 * \
	rfm_df['F_rank_norm']+0.57*rfm_df['M_rank_norm']
rfm_df['RFM_Score'] *= 0.05
rfm_df = rfm_df.round(2)
rfm_df[['CustomerID', 'RFM_Score']].head(7)
rfm_df['RFM_Score'].unique()
"""
Rating Customer based upon the RFM score
rfm score > 3.5 : Champions
3.5 > rfm score > 3 : Potential customers 
3 > rfm score > 2.5 : Need Attention
2.5 > rfm score : Lost customer

"""
rfm_df["Customer_segment"] = np.where(rfm_df['RFM_Score'] >
								3.5, "Champions",
									(np.where(
										rfm_df['RFM_Score'] > 3,
										"Potential Customers",
										(np.where(
                                            rfm_df['RFM_Score'] > 2.5,
							'Needed Attention','Lost Customers')))))
rfm_df[['CustomerID', 'RFM_Score', 'Customer_segment']].head(20)


plt.pie(rfm_df.Customer_segment.value_counts(),
		labels=rfm_df.Customer_segment.value_counts().index,
		autopct='%.0f%%')
plt.show()

"""
Inference :
    58% of the total customers are Potential Customers, 
    22% of the total customers are customers who need attention,
    21% of the total customers are Champions.
    So the company should focus more on 22% of the total customers who need 
    attention and give them some offer to encourge them for shopping.

