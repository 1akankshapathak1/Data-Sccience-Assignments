"""
Build a recommender system by using cosine simillarties score.

"""
import pandas as pd
import seaborn as sns
data=pd.read_csv('E:\\assignments\\Recommendation System\\book.csv',encoding='latin1')
pd.set_option('display.max_colwidth', -1)
data.shape
list(data)
data.head()
data.dtypes
len(data)
data['Book.Title'].value_counts()

%matplotlib qt
sns.countplot(data['Book.Title'])
sns.distplot(data['Book.Rating'])

data.sort_values('User.ID')
len(data['User.ID'].unique())
data['User.ID'].describe()
data['Book.Rating'].hist()
data['Book.Rating'].value_counts()

book_df=data.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')
book_df.iloc[0]
list(book_df)

#To fill all NaN values with zeroes(0)
book_df.fillna(0,inplace=True)
book_df

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
#Cosine similarity score
cos=1-pairwise_distances(book_df.values,metric='cosine') 

cos.shape
cos_df=pd.DataFrame(cos)

cos_df.index=data['User.ID'].unique()
cos_df.columns=data['User.ID'].unique()

cos_df.iloc[0:7,0:7]

cos_df.max()
cos_df.idxmax(axis=1)[0:7]

#Correlation score
corr=1-pairwise_distances(book_df.values, metric='correlation') #Correlation similarity score
corr.shape
corr_df=pd.DataFrame(corr)

corr_df.index=data['User.ID'].unique()
corr_df.columns=data['User.ID'].unique()

corr_df.iloc[0:7,0:7]

corr_df.max()
corr_df.idxmax(axis=1)[0:5]

'''
Inference: As there are multiple books and there are many users who have read only
           one book and very few have read more than one. 
           There is only one book which have atmost of 5 readers and by this
           we can understand that there are large number of NaN or Zeroes
           as similarity scores, So it is difficult to recommend the books
           to the readers with the given dataset.
'''
