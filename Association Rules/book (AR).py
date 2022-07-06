"""
Prepare rules for the all the data sets 
1) Try different values of support and confidence. 
Observe the change in number of rules for different support,confidence values
2) Change the minimum length in apriori algorithm
3) Visulize the obtained rules using different plots 
"""

import pandas as pd
df=pd.read_csv('E:\\assignments\\Association Rules\\book.csv')
df.shape
list(df)
df.head()
df.info()
df.values
type(df.values)

book=pd.get_dummies(df)
book.shape    
list(book)

#import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules 
#from mlxtend.preprocessing import TransactionEncoder   

freq_items=apriori(book,min_support=0.1,use_colnames=True)  
freq_items

asr=association_rules(freq_items,metric='lift',min_threshold=0.6) 
asr
asr.sort_values('lift',ascending=False)
asr.sort_values('lift',ascending=False)[0:20]   

asr[asr.lift>1]
asr[['support','confidence','lift']].hist() 

%matplotlib qt
import matplotlib.pyplot as plt
plt.scatter(asr['support'], asr['confidence'])
plt.show()    

import seaborn as sns
sns.scatterplot('support', 'confidence', data=asr, hue='antecedents')
plt.show()
   
