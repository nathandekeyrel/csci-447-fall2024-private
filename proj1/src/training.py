import numpy as np


class NaiveBayes:
    def __init__(self):
        """
        The Naive Bayes classifier implementation.

        Attributes:
            classes (numpy.ndarray): Unique classes in the training data.
            class_probs (dict): Prior probabilities of each class.
            feature_probs (dict): Conditional probabilities of features given each class.
        """
        self.classes = None
        self.d = 0
        self.num_per_class = {}
        self.class_probs = {}  # Q(C = c_i)
        self.feature_probs = {}  # F(A_j = a_k, C = c_i)

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier on the given data.

        Calculates the prior probabilities of each class and the
        conditional probabilities of each feature given each class.

        :param X: numpy.ndarray, shape (n_samples, n_features)
            The training input samples.
        :param y: numpy.ndarray, shape (n_samples,)
            The target values (class labels).
        """
        self.classes = np.unique(y)
        N = len(y)
        d = X.shape[1]  # number of attributes
        self.d = d

        # calc Q(C = c_i) for each class
        for c in self.classes:
            self.num_per_class[c] = np.sum(y == c)
            self.class_probs[c] = self.num_per_class[c] / N

        # calc F(A_j = a_k, C = c_i) for each class and attribute
        for c in self.classes:
            X_c = X[y == c]
            N_c = len(X_c)
            self.feature_probs[c] = {}

            for j in range(d):
                self.feature_probs[c][j] = {}
                unique_values, counts = np.unique(X_c[:, j], return_counts=True)
                for a_k, count in zip(unique_values, counts):
                    self.feature_probs[c][j][a_k] = (count + 1) / (N_c + d)

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        :param X: numpy.ndarray, shape (n_samples, n_features)
            The input samples to classify.
        :return: numpy.ndarray, shape (n_samples,)
            Predicted class labels for each input sample.
        """
        prediction = np.array([self._predict_single(x) for x in X])
        return prediction

    def _predict_single(self, x):
        """
        Predict the class label for a single input sample.

        Calculate the posterior probability for each class given the
        input features and returns the class with the highest probability.

        :param x: numpy.ndarray, shape (n_features,)
            A single input sample to classify.
        :return: object
            Predicted class label for the input sample.
        """
        class_scores = {}
        for c in self.classes:
            score = np.log(self.class_probs[c])  # log to prevent underflow
            for j, a_k in enumerate(x):
                if a_k in self.feature_probs[c][j]:
                    score += np.log(self.feature_probs[c][j][a_k])
                else:
                    score += np.log(1 / (self.d + self.num_per_class[c]))  # laplace smoothing
            class_scores[c] = score

        max_score = max(class_scores, key=class_scores.get)
        return max_score
