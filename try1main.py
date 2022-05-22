import numpy as np
import matplotlib.pyplot as pyPlot
import pandas as pd

def update_assignments(data, centroids):
      c = []
      for i in data:
           c.append(np.argmin(np.sum((i.reshape((1, 2)) - centroids) ** 2, axis = 1)))
      return c

def update_centroids(data, num_clusters, assignments):
      cen = []
      for c in range(len(num_clusters)):
            cen.append(np.mean([data[x] for x in range(len(data)) if assignments[x] == c], axis = 0 ))
      return cen

#data = np.read_csv('anime.csv').T
data = pd.read_csv('anime.csv').T
print(data.shape)

centroids = (np.random.normal(size = (3,14578)) * 0.0001) +np.mean(data, axis = 0).reshape((1,2))
for i in range(100):
      a = update_assignments(data, centroids)
      centroids = update_centroids(data, centroids, a)
      centroids = np.array(centroids)

pyPlot.scatter(data[:, 0], data[:, 1])
pyPlot.scatter(centroids[:, 0], centroids[:, 1])

pyPlot.show()