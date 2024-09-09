import numpy as np


class NaiveBayes:
    def __init__(self):
        """

        """
        self.classes = None
        self.class_probabilities = {}  # --> Q(C = c_i)
        self.feature_probabilities = {}  # --> F(A_i = a_k, C = c_i)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            self.class_probabilities[c] = np.sum(y == c) / n_samples

        for c in self.classes:
            self.feature_probabilities[c] = {}
            X_c = X[y == c]
            N_c = len(X_c)
            for j in range(n_features):
                self.feature_probabilities[c][j] = {}
                unique_values, counts = np.unique(X_c[:, j], return_counts=True)
                for a_k, count in zip(unique_values, counts):
                    self.feature_probabilities[c][j][a_k] = (count + 1) / (N_c + n_features)

        return self

    def predict(self, X):
        """
        :param X: input
        :param X:
        :return:
        """
        prediction = np.array([self._predict_single(x) for x in X])
        return prediction

    def _predict_single(self, x):
        probabilities = {}
        for c in self.classes:
            probability = np.log(self.class_probabilities[c])
            for feature, value in enumerate(x):
                if value in self.feature_probabilities[c][feature]:
                    probability += np.log(self.feature_probabilities[c][feature][value])
                else:
                    probability += np.log(1e-10)
            probabilities[c] = probability

        max_probability = max(probabilities, key=probabilities.get)
        return max_probability
