# -*- coding: utf-8 -*-
"""
Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and
 identify the number of clusters formed and draw inferences.

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("E:\\assignments\\Clustering\\crime_data.csv")

X = df.iloc[:,1:5].values
X

# Hierarchical Clustering
#------------------------
# Dendogram
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X, method='ward'))

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(X, method='ward'))
plt.axhline(y=400, color='r', linestyle='--')

# the line cuts the dendogram at two points, so we have 2 clusters.

# Model Fitting 
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
Y = cluster.fit_predict(X)

plt.figure(figsize=(16,9))
plt.scatter(X[:,0],X[:,1],X[:,2],c=Y,cmap='rainbow')
#==============================================================================
# K means Clustering
#-------------------

from sklearn.cluster import KMeans
wcss = [] 
for i in range(1, 11): 
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    km.fit_predict(X) 
    wcss.append(km.inertia_)
    wcss

 # Elbow plot   
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

# The elbow shape is created at 5, i.e., our K value or an optimal number of clusters is 5.

# Model Fitting

km = KMeans(n_clusters = 5)
Y_means = km.fit_predict(X) 
Y_means

c=km.cluster_centers_
km.inertia_

# Plotting the result
%matplotlib qt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X[Y_means == 0,0],X[Y_means == 0,1],X[Y_means==0,2],X[Y_means == 0,3],color='blue')
ax.scatter(X[Y_means == 1,0],X[Y_means == 1,1],X[Y_means==1,2],X[Y_means == 1,3],color='red')
ax.scatter(X[Y_means == 2,0],X[Y_means == 2,1],X[Y_means==2,2],X[Y_means == 2,3],color='orange')
ax.scatter(X[Y_means == 3,0],X[Y_means == 3,1],X[Y_means==3,2],X[Y_means == 3,3],color='green')
ax.scatter(X[Y_means == 4,0],X[Y_means == 4,1],X[Y_means==4,2],X[Y_means == 4,3],color='black')
#==============================================================================
# DBScan
#--------
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=5,min_samples=5)
labels = db.fit_predict(X)
np.unique(labels)
# we are getting unique value as -1 which means it considers all data points as noise

"""
Inference : 
Hierarchical Clustering - the line cuts the dendogram at two points, so 2 clusters are formed.
 
K means Clustering - The elbow shape is created at 5, i.e., our K value or an optimal number of clusters is 5.
                      with inertia 24417.023523809516
                      
DBScan - we are getting unique value as -1 which means it considers all data point as noise                      
         so no cluser is formed.
"""












