import numpy as np


class KMeansClassification:
    """K-Means clustering algorithm for classification tasks.

    This class implements a modified K-Means clustering algorithm that can be used
    for classification.
    """

    def __init__(self, n_clusters: int, max_iterations=100):
        """Initialize the KMeans object for classification.

        :param n_clusters: The number of clusters to form.
        :param max_iterations: The maximum number of iterations for the K-Means algorithm.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels_ = None
        self.cluster_points = None
        self.cluster_labels = None

    @staticmethod
    def euclidian_distance(x1, x2):
        """Calculate the Euclidean distance between two vectors.

        :param x1: First vector
        :param x2: Second vector
        :return: The Euclidean distance between x1 and x2
        """
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def fit(self, X, y):
        """Perform K-Means clustering on the input data and store cluster information.

        :param X: Input data to cluster, shape (n_samples, n_features)
        :param y: Target labels corresponding to each sample in X
        :return: The fitted KMeansClassification object
        """
        # initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters)]

        for _ in range(self.max_iterations):
            # calculate distances between each point and all centroids
            distances = np.array([[self.euclidian_distance(x, c) for c in self.centroids] for x in X])
            self.labels_ = np.argmin(distances, axis=1)  # assign each point to nearest centroid

            # update centroid based on mean of points in each cluster
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) if np.sum(self.labels_ == i) > 0
                                      else self.centroids[i] for i in range(self.n_clusters)])

            # check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # store points and labels for clusters
        self.cluster_points = [X[self.labels_ == i] for i in range(self.n_clusters)]
        self.cluster_labels = [y[self.labels_ == i] for i in range(self.n_clusters)]

        return self

    def predict(self, X, k=3):
        """Predict class labels for the input data using a combination of K-Means and KNN.

        :param X: Input data to classify, shape (n_samples, n_features)
        :param k: Number of nearest neighbors to consider for KNN within each cluster
        :return: Predicted class labels for each data point
        """
        predictions = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            # find index of the nearest centroid
            distances = np.array([self.euclidian_distance(x, c) for c in self.centroids])
            nearest_cluster = np.argmin(distances)

            # retrieve data points from X
            cluster_X = self.cluster_points[nearest_cluster]
            # retrieve corresponding labels
            cluster_y = self.cluster_labels[nearest_cluster]

            # Handle empty clusters
            if len(cluster_X) == 0:
                # If the nearest cluster is empty, use the second nearest
                distances[nearest_cluster] = np.inf
                nearest_cluster = np.argmin(distances)
                cluster_X = self.cluster_points[nearest_cluster]
                cluster_y = self.cluster_labels[nearest_cluster]

            # adjust k if cluster has fewer than k points
            if len(cluster_X) < k:
                k = max(1, len(cluster_X))

            # calculate euclidian distance between input points and each point in the cluster
            distances = np.array([self.euclidian_distance(x, cx) for cx in cluster_X])
            # sort distances in ascending order and returns indices of k smallest distances
            nearest_indices = np.argsort(distances)[:k]
            # retrieves labels of k nearest neighbor using indices
            nearest_labels = cluster_y[nearest_indices]

            # majority voting
            if len(nearest_labels) > 0:
                predictions[i] = np.argmax(np.bincount(nearest_labels))

        return predictions

    def get_reduced_dataset(self):
        """Get the reduced dataset consisting of centroids and their corresponding majority labels.

        :return: A tuple of 2 numpy arrays (centroids, labels)
        """
        majority_labels = np.array([np.argmax(np.bincount(cluster_labels)) if len(cluster_labels) > 0
                                    else -1 for cluster_labels in self.cluster_labels])
        return self.centroids, majority_labels


class KMeansRegression:
    """K-Means clustering algorithm for regression.

    This class implements the K-Means clustering algorithm, which turns n observations
    into k clusters where each observation belongs to the cluster with the nearest centroid.
    """

    def __init__(self, n_clusters: int, max_iterations=100):
        """Initialize the KMeans object.

        :param n_clusters: The number of clusters to form.
        :param max_iterations: The maximum number of iterations for the algorithm.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.cluster_targets = None
        self.centroids = None
        self.labels_ = None

    @staticmethod
    def euclidian_distance(x1, x2):
        """Calculate the Euclidean distance between two vectors.

        :param x1: First vector
        :param x2: Second vector
        :return: The Euclidean distance between x1 and x2
        """
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def fit(self, X, y):
        """Perform K-Means clustering on the input data.

        :param self: The k-means object
        :param X: Input data to cluster, shape (n_samples, n_features)
        :param y: Target values corresponding to each sample in X
        :return: The fitted KMeans object
        """
        # initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters)]

        for _ in range(self.max_iterations):
            # calculate distances between each point and all centroids
            distances = np.array([[self.euclidian_distance(x, c) for c in self.centroids] for x in X])
            self.labels_ = np.argmin(distances, axis=1)  # assign each point to nearest centroid

            # update centroid based on mean of points in each cluster
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) if np.sum(self.labels_ == i) > 0
                                      else self.centroids[i] for i in range(self.n_clusters)])

            # check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # calculate cluster targets, handle empty clusters
        self.cluster_targets = np.array([y[self.labels_ == i].mean() if np.sum(self.labels_ == i) > 0
                                         else np.nan for i in range(self.n_clusters)])
        return self

    def predict(self, X, k=None, sign=None):
        """Predict cluster labels for the input data.

        :param self: The k-means object
        :param X: Input data to label, shape (n_samples, n_features)
        :return: Predicted cluster labels for each data point
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted. Call 'fit' before 'predict'.")

        # calculate distance between each point and all centroids
        distances = np.array([[self.euclidian_distance(x, c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)  # assign each point to nearest centroid

    def get_reduced_dataset(self):
        """Get the reduced dataset consisting of centroids and their corresponding mean target values.

        :return: A tuple of 2 numpy arrays
        """
        if self.centroids is None or self.cluster_targets is None:
            raise ValueError("Model has not been fitted. Call 'fit' before getting reduced dataset.")

        # remove centroids and targets in empty clusters
        valid_indices = ~np.isnan(self.cluster_targets)
        return self.centroids[valid_indices], self.cluster_targets[valid_indices]
