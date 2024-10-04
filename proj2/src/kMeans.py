import numpy as np


class KMeans:
    """ K-Means clustering algorithm implementation.

    This class implements the K-Means clustering algorithm, which turns n observations
    into k clusters where each observation belongs to the cluster with the nearest centroid.
    """

    def __init__(self, n_clusters: int, max_iterations=100):
        """ Initialize the KMeans object.

        :param n_clusters: The number of clusters to form.
        :param max_iterations: The maximum number of iterations for the algorithm.
        """
        self.n_clusters = n_clusters
        self.cluster_targets = None
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels_ = None

    @staticmethod
    def euclid_dist(x1, x2):
        """ Calculate the Euclidean distance between two vectors.

        :param x1: First vector
        :param x2: Second vector
        :return: The Euclidean distance between x1 and x2
        """
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def fit(self, X, y):
        """ Perform K-Means clustering on the input data.

        :param X: Input data to cluster, shape (n_samples, n_features)
        :param y: Target values corresponding to each sample in X
        :return: The fitted KMeans object
        """
        # initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iterations):
            # calculate distances between each point and all centroids
            distances = np.array([[self.euclid_dist(x, c) for c in self.centroids] for x in X])
            labels = np.argmin(distances, axis=1)  # assign each point to nearest centroid

            # update centroid based on mean of points in each cluster
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.cluster_targets = [y[self.labels_ == i].mean() for i in range(self.n_clusters)]
        return self

    def predict(self, X):
        """ Predict cluster labels for the input data.

        :param X: Input data to label, shape (n_samples, n_features)
        :return: Predicted cluster labels for each data point
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted. Call 'fit' before 'predict'.")

        # calculate distance between each point and all centroids
        distances = np.array([[self.euclid_dist(x, c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)  # assign each point to nearest centroid

    def fit_predict(self, X, y):
        """ Perform K-Means clustering and return cluster labels.

        :param X: Input data to cluster, shape (n_samples, n_features)
        :param y: Target values corresponding to each sample in X
        :return: Cluster labels for each data point
        """
        return self.fit(X, y).predict(X)

    def get_centroids(self):
        """ Get the centroids of the clusters.

        :return: Array of centroids, shape (n_clusters, n_features)
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted. Call 'fit' before getting centroids.")
        return self.centroids

    def get_reduced_dataset(self):
        """ Get the reduced dataset consisting of centroids and their corresponding mean target values.

        :return: List of lists, each containing a centroid and its mean target value
        """
        if self.centroids is None or self.cluster_targets is None:
            raise ValueError("Model has not been fitted. Call 'fit' before getting reduced dataset.")

        targets = [list(centroid) + [target] for centroid, target in zip(self.centroids, self.cluster_targets)]
        return targets
