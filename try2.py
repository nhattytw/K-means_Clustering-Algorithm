import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def euclideanDistance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=5, maxIterations=100, plotSteps=False):
        self.K = K
        self.maxIterations = maxIterations
        self.plotSteps = plotSteps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        randomSampleIndexes = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in randomSampleIndexes]

        # Optimize clusters
        for _ in range(self.maxIterations):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self.createClusters(self.centroids)

            if self.plotSteps:
                self.plot()

            # Calculate new centroids from the clusters
            centroidsOld = self.centroids
            self.centroids = self.getCentroids(self.clusters)

            # check if clusters have changed
            if self.isConverged(centroidsOld, self.centroids):
                break

            if self.plotSteps:
                self.plot()

        # Classify samples as the index of their clusters
        return self.getClusterLabels(self.clusters)

    def getClusterLabels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        for clusterIndexes, cluster in enumerate(clusters):
            for sampleIndex in cluster:
                labels[sampleIndex] = clusterIndexes
        return labels

    def createClusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closestCentroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def closestCentroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclideanDistance(sample, point) for point in centroids]
        closestIndex = np.argmin(distances)
        return closestIndex

    def getCentroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for clusterIndexes, cluster in enumerate(clusters):
            clusterMean = np.mean(self.X[cluster], axis=0)
            centroids[clusterIndexes] = clusterMean
        return centroids

    def isConverged(self, centroidsOld, centroids):
        # distances between each old and new centroids, fol all centroids
        distances = [
            euclideanDistance(centroidsOld[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


# Testing
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=2, n_samples=10000, n_features=2, shuffle=True, random_state=100
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, maxIterations=250, plotSteps=True)
    y_pred = k.predict(X)

    k.plot()