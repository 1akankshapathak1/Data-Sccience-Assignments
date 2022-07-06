
"""
Perform clustering (hierarchical,K means clustering and DBSCAN) for the 
airlines data to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained.

"""
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel("E:\\assignments\\Clustering\\EastWestAirlines.xlsx",sheet_name='data')
df.head()
df.shape
list(df)
df.dtypes
df.drop(['ID#'],axis=1,inplace=True)

X=df.iloc[:,1:12].values
X.shape
list(X)

# Data Transformation
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_scale=SS.fit_transform(X)
#==============================================================================
# Hierarchical Clustering
# Dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X_scale, method='ward'))

# Number of clusters
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
z = shc.linkage(X_scale, method='ward')
dend = shc.dendrogram(z)
plt.axhline(y=80,color='r', linestyle='--')
plt.show()

# the line cuts the dendogram at 6 points, so we have 6 clusters

# Model Fitting 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X_scale)

# Plotting resulting clusters

# Function for creating datapoints in the form of a circle
import matplotlib
import math
def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30,30),math.sin(2*math.pi/n*x)*r+np.random.normal(-30,30)) for x in range(1,n+1)]

# Creating data points in the form of a circle
X_scale=pd.DataFrame(PointsInCircum(2000,3000))
X_scale=X_scale.append(PointsInCircum(800,1800))
X_scale=X_scale.append(PointsInCircum(100,800))
# Adding noise to the dataset
X_scale=X_scale.append([(np.random.randint(-600,600),np.random.randint(-600,600)) for i in range(300)])

plt.figure(figsize=(10,10))
plt.scatter(X_scale[0],X_scale[1],c=Y,cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('Hierarchical Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

#=================================================================================
# K means clustering
#-------------------
from sklearn.cluster import KMeans
wcss = [] 
for i in range(1, 11): 
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 1)
    km.fit_predict(X_scale) 
    wcss.append(km.inertia_)
    wcss

 # Elbow plot   
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

# The elbow shape is created at 8, i.e., our K value or an optimal number of clusters is 8.

# Model Fitting
km = KMeans(n_clusters = 8)
Y_means = km.fit_predict(X_scale) 
Y_means

c=km.cluster_centers_
km.inertia_

# Plotting the result
colors=['purple','red','blue','green']
plt.figure(figsize=(10,10))
plt.scatter(X_scale[0],X_scale[1],c=Y_means,cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('K-Means Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

#==============================================================================
# DBScan
#--------
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=5,min_samples=5)
labels = db.fit_predict(X_scale)

cl=pd.DataFrame(db.labels_,columns=['Cluster'])
cl
cl['Cluster'].value_counts()

data_new=pd.concat([pd.DataFrame(X_scale),cl],axis=1)

#Noise data
nd=data_new[data_new['Cluster']==-1]
nd
nd.shape

#Final data without outliers
fd1 = data_new[data_new['Cluster']==0]
fd2 = data_new[data_new['Cluster']==1]
fd3 = data_new[data_new['Cluster']==2]
fd4 = data_new[data_new['Cluster']==3]
fd = pd.concat([fd1,fd2,fd3,fd4],axis=0)
fd.shape
data_new.mean()
fd.mean()



plt.figure(figsize=(10,10))
plt.scatter(X_scale[0],X_scale[1],s=15,color='grey')
plt.title('DBScan',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

"""
Inference: 
 
 Hierarchical Clustering - the line cuts the dendogram at 6 points, so we have 6 clusters

K means - The elbow shape is created at 8, i.e., our K value or an optimal number of clusters is 8.
          
DBScan - 4 clusters are formed

In Hierarchical and K menas failed to cluster the data points properly.
While DBScan clusters the data points properly.
Hence, DBscan is the best model for this data.    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
