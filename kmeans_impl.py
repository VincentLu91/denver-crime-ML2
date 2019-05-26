# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
df = pd.read_csv("crime.csv")
X = df.iloc[:, [15,2]].values
# consider precinct IDs as X coordinates, offense code as Y coordinates
# X column - 15
# Y column - 2

from sklearn.cluster import KMeans

wcss = []
for i in range (1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300,
                    n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# optimal number of clusters - 2

kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10,
                max_iter = 300, random_state = 0)
kmeans.fit(X)
y_clusters = kmeans.predict(X)
print(kmeans.cluster_centers_[:,0])
print(kmeans.cluster_centers_[:,1])

plt.scatter(X[y_clusters == 0, 0], X[y_clusters == 0, 1], s = 600,
            c = 'red', label = 'Cluster 1')
plt.scatter(X[y_clusters == 1, 0], X[y_clusters == 1, 1], s = 600,
            c = 'blue', label = 'Cluster 2')
#plt.scatter(X[y_clusters == 2, 0], X[y_clusters == 2, 1], s = 600,
#            c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s = 500, c = 'yellow', label = 'Centroids')
