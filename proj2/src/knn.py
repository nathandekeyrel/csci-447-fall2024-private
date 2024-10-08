import numpy as np
import heapq as hq


class KNNClassifier:
    #init with the data D
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X, k):
        return np.array([self._predict(x, k) for x in X])

    def _predict(self, x, k):
        distances = [euclidianDistance(x, xt) for xt in self.X]
        indices = np.argsort(distances)[:k]
        votes = {}
        for index in indices:
            if self.y[index] in votes:
                votes[self.y[index]] += 1
            else:
                votes.update({self.y[index] : 1})
        votes = list(votes.items())
        votes.sort(key = lambda x : x[1])
        return votes[0][0]

class KNNRegression:
    #init with the data D
    def __init__(self):
        pass
    
    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X, k, sig=10):
        return np.array([self._predict(x, k, sig) for x in X])

    def _predict(self, x, k, sig):
        distances = [euclidianDistance(x, xt) for xt in self.X]
        indices = np.argsort(distances)[:k]
        weights = [RBFkernel(distances[i], sig) for i in indices]
        w = sum(weights)
        nns = [self.Y[i] for i in indices]
        s = sum([nns[i] * weights[i] for i in range(len(indices))])
        prediction = s / w
        return prediction



#radial basis function kernel
def RBFkernel(distance, sig):
    return np.exp(-(distance * distance) / (2 * sig * sig))

# euclidian distance algorithm
def euclidianDistance(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.sum(diff * diff))
