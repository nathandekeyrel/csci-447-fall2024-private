import numpy as np


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_probs = {}  # Q(C = c_i)
        self.feature_probs = {}  # F(A_j = a_k, C = c_i)

    def fit(self, X, y):
        self.classes = np.unique(y)
        N = len(y)
        d = X.shape[1]  # number of attributes

        # Step 1: Calculate Q(C = c_i) for each class
        for c in self.classes:
            self.class_probs[c] = np.sum(y == c) / N

        # Step 2 & 3: Calculate F(A_j = a_k, C = c_i) for each class and attribute
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
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        class_scores = {}
        for c in self.classes:
            score = np.log(self.class_probs[c])  # Using log for numerical stability
            for j, a_k in enumerate(x):
                if a_k in self.feature_probs[c][j]:
                    score += np.log(self.feature_probs[c][j][a_k])
                else:
                    score += np.log(1 / (len(self.feature_probs[c][j]) + 1))  # Laplace smoothing
            class_scores[c] = score

        return max(class_scores, key=class_scores.get)