import numpy as np


class KMeans:
    def __init__(self, n_clusters: int, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None

    @staticmethod
    def euclid_dist(x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)

    def kmeans(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iterations):
            distances = np.array([[self.euclid_dist(x, c) for c in self.centroids] for x in X])
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return labels

    def get_labels(self, X):
        distances = np.array([[self.euclid_dist(x, c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)

    def centroids(self):
        return self.centroids
