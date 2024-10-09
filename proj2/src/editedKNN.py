import numpy as np
import evaluating as ev
import sys


class EKNNErrClassifier:
    """Edited K-Nearest Neighbors Classifier using error-based editing.
    This classifier edits the training set to improve classification accuracy.
    """

    def __init__(self):
        """
        Initialize the eKNN classifier
        """
        self.X = None
        self.y = None
        self.mark = None
        self.size = None

    def fit(self, X, Y):
        """Fit the classifier with training data.

        Parameters:
        X (array-like): Training feature vectors
        Y (array-like): Training labels
        """
        self.X = X
        self.y = Y
        self.size = len(self.X)
        self.mark = np.repeat([True], self.size)

    def edit(self, Xt, Yt):
        """Edit the training set based on classification errors.

        Parameters:
        Xt (array-like): Test feature vectors
        Yt (array-like): Test labels
        """
        self.mark = np.repeat([True], self.size)
        q = quality(self, Xt, Yt)  # quality before edit
        qd = q  # quality after edit
        while qd >= q:
            changed = False
            # iterate through all training examples
            for i in range(self.size):
                if self.mark[i]:
                    self.mark[i] = False
                    # check if the example is misclassified
                    if not self._predict(self.X[i], 1) == self.y[i]:
                        if not changed:
                            changed = True
                    else:
                        self.mark[i] = True
            if not changed:
                return
            q = qd
            qd = quality(self, Xt, Yt)

    def predict(self, X, k):
        """Predict labels for multiple input samples.

        Parameters:
        X (array-like): Input samples
        k (int): Number of neighbors to consider

        Returns:
        array: Predicted labels
        """
        return np.array([self._predict(x, k) for x in X])

    def _predict(self, x, k):
        """Predict label for a single input sample.

        Parameters:
        x (array-like): Input sample
        k (int): Number of neighbors to consider

        Returns:
        object: Predicted label
        """
        distances = []
        # calculate distances to all training examples
        for i in range(len(self.X)):
            if not self.mark[i]:
                distances.append(sys.float_info.max)
                continue
            distances.append(euclidianDistance(x, self.X[i]))

        # find k nearest neighbors
        indices = np.argsort(distances)[:k]

        # count votes for each class
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

    def numberOfExamples(self):
        """Get the number of examples remaining after editing.

        Returns:
        int: Number of marked examples
        """
        return self.mark.sum()


class EKNNErrRegression:
    """
    Edited K-Nearest Neighbors Regression using error-based editing.
    This regressor edits the training set to improve prediction accuracy.
    """

    def __init__(self):
        """
        Initialize the eKNN regressor
        """
        self.X = None
        self.y = None
        self.size = None
        self.mark = None

    def fit(self, X, Y):
        """Fit the regressor with training data.

        Parameters:
        X (array-like): Training feature vectors
        Y (array-like): Training target values
        """
        self.X = X
        self.y = Y
        self.size = len(self.X)
        self.mark = np.repeat([True], self.size)

    def edit(self, Xt, Yt, sig, e):
        """Edit the training set based on prediction errors.

        Parameters:
        Xt (array-like): Test feature vectors
        Yt (array-like): Test target values
        sig (float): Sigma parameter for RBF kernel
        e (float): Error threshold for editing
        """
        self.mark = np.repeat([True], self.size)
        q = qualityR(self, Xt, Yt, sig)
        qd = q
        while qd <= q:
            changed = False
            # iterate through all training examples
            for i in range(len(self.X)):
                if self.mark[i]:
                    self.mark[i] = False
                    # check if the prediction error exceeds the threshold
                    if np.abs(self._predict(self.X[i], 1, 1) - self.y[i]) >= e:
                        if not changed:
                            changed = True
                    else:
                        self.mark[i] = True
            if not changed:
                return
            q = qd
            qd = qualityR(self, Xt, Yt, sig)

    def predict(self, X, k, sig):
        """Predict target values for multiple input samples.

        Parameters:
        X (array-like): Input samples
        k (int): Number of neighbors to consider
        sig (float): Sigma parameter for RBF kernel

        Returns:
        array: Predicted target values
        """
        return np.array([self._predict(x, k, sig) for x in X])

    def _predict(self, x, k, sig):
        """Predict target value for a single input sample.

        Parameters:
        x (array-like): Input sample
        k (int): Number of neighbors to consider
        sig (float): Sigma parameter for RBF kernel

        Returns:
        float: Predicted target value
        """
        distances = []
        # calculate distances to all training examples
        for i in range(len(self.X)):
            if not self.mark[i]:
                distances.append(sys.float_info.max)
                continue
            distances.append(euclidianDistance(x, self.X[i]))

        # find k nearest neighbors
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

    def numberOfExamples(self):
        """Get the number of examples remaining after editing.

        Returns:
        int: Number of marked examples
        """
        return self.mark.sum()


def RBFkernel(distance, sig):
    """Radial Basis Function (RBF) kernel.

    Parameters:
    distance (float): Distance between two points
    sig (float): Sigma parameter for RBF kernel

    Returns:
    float: RBF kernel value
    """
    return np.exp(-(distance * distance) / (2 * sig * sig))


def euclidianDistance(x1, x2):
    """Calculate Euclidean distance between two points.

    Parameters:
    x1 (array-like): First point
    x2 (array-like): Second point

    Returns:
    float: Euclidean distance between x1 and x2
    """
    diff = x1 - x2
    return np.sqrt(np.sum(diff * diff))


def quality(M, Xt, Yt):
    """Calculate the quality (accuracy) of the classifier on test data.

    Parameters:
    M: Classifier model
    Xt (array-like): Test feature vectors
    Yt (array-like): Test labels

    Returns:
    float: Accuracy of the classifier
    """
    pred = M.predict(Xt, 1)
    r = 1 - ev.zero_one_loss(Yt, pred)
    return r


def qualityR(M, Xt, Yt, sig=10):
    """Calculate the quality (mean absolute error) of the regression model on test data.

    Parameters:
    M: Regression model
    Xt (array-like): Test feature vectors
    Yt (array-like): Test target values
    sig (float): Sigma parameter for RBF kernel (default: 10)

    Returns:
    float: Mean absolute error of the regression model
    """
    e = 0
    # calculate absolute error for each test sample
    for x, y in zip(Xt, Yt):
        ei = M._predict(x, 1, sig) - y
        e += np.abs(ei)
    # return mean absolute error
    return e / len(Xt)
