import numpy as np


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implementation.
    """

    def __init__(self):
        """
        Initialize the KNN Classifier.
        """
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Fit the KNN Classifier on the training data.

        :param X: Training feature vectors
        :param y: Training labels
        """
        self.X = X
        self.y = y

    def predict(self, X, k):
        """
        Predict labels for multiple input samples.

        :param X: Input samples to predict
        :param k: Number of neighbors to consider
        :return: Predicted labels for input samples
        """
        return np.array([self._predict(x, k) for x in X])

    def _predict(self, x, k):
        """Predict label for a single input sample.

        :param x: Input sample to predict
        :param k: Number of neighbors to consider
        :return: Predicted label for the input sample
        """
        # calculate distances between input sample and all training samples
        distances = [euclidianDistance(x, xt) for xt in self.X]
        # get indices of k nearest neighbors
        indices = np.argsort(distances)[:k]
        # count votes for each class among k nearest neighbors
        votes = {}
        for index in indices:
            if self.y[index] in votes:
                votes[self.y[index]] += 1
            else:
                votes.update({self.y[index]: 1})
        # sort votes and return the class with the most votes
        votes = list(votes.items())
        votes.sort(key=lambda x: x[1], reverse=True)
        return votes[0][0]


class KNNRegression:
    """
    K-Nearest Neighbors Regression implementation.
    """

    def __init__(self):
        """
        Initialize the KNN Regression model.
        """
        self.X = None
        self.y = None

    def fit(self, X, y):
        """Fit the KNN Regression model on the training data.

        :param X: Training feature vectors
        :param y: Training target values
        """
        self.X = X
        self.y = y

    def predict(self, X, k, sig=10):
        """Predict target values for multiple input samples.

        :param X: Input samples to predict
        :param k: Number of neighbors to consider
        :param sig: Sigma parameter for RBF kernel (default: 10)
        :return: Predicted target values for input samples
        """
        return np.array([self._predict(x, k, sig) for x in X])

    def _predict(self, x, k, sig):
        """Predict target value for a single input sample.

        :param x: Input sample to predict
        :param k: Number of neighbors to consider
        :param sig: Sigma parameter for RBF kernel
        :return: Predicted target value for the input sample
        """
        # calculate distances between input sample and all training samples
        distances = [euclidianDistance(x, xt) for xt in self.X]
        # get indices of k nearest neighbors
        indices = np.argsort(distances)[:k]
        # calculate weights using RBF kernel
        weights = [RBFkernel(distances[i], sig) for i in indices]
        w = sum(weights)
        # get target values of k nearest neighbors
        nns = [self.y[i] for i in indices]
        # calculate weighted sum of target values
        s = sum([nns[i] * weights[i] for i in range(len(indices))])
        # return weighted average as prediction
        prediction = s / w
        return prediction


def RBFkernel(distance, sig):
    """Radial Basis Function (RBF) kernel.

    :param distance: Distance between two points
    :param sig: Sigma parameter for RBF kernel
    :return: RBF kernel value
    """
    return np.exp(-(distance * distance) / (2 * sig * sig))


def euclidianDistance(x1, x2):
    """Calculate Euclidean distance between two points.

    :param x1: First point
    :param x2: Second point
    :return: Euclidean distance between x1 and x2
    """
    diff = x1 - x2
    return np.sqrt(np.sum(diff * diff))
