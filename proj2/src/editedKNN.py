import knn
import heapq as hq
import numpy as np
import copy

class EKNNErrClassifier:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.size = len(self.X)
        self.mark = np.repeat([True], self.size)

    def edit(self, Xt, Yt):
        q = quality(self, Xt, Yt) # quality before edit
        qd = q # quality after edit
        while qd >= q: 
            changed = False
            for i in range(self.size):
                self.mark[i] = False
                if not self._predict(self.X[i]) == self.Y[i]:
                    if not changed:
                        changed = True
                else:
                    self.mark[i] = True
            if not changed:
                return
            q = qd
            qd = quality(self.cl, Xt, Yt)

    def predict(self, X, k):
        return [self._predict(x, k) for x in X]

    def _predict(self, x, k):
        return self.cl._predict(x, k)
    
    def numberOfExamples(self):
        return self.mark.sum()

class EKNNErrRegression:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.mark = np.repeat([True], len(self.X))

    def edit(self, Xt, Yt, sig, e):
        self.mark = np.repeat([True], len(self.X))
        q = qualityR(self, Xt, Yt, sig)
        qd = q
        while qd <= q:
            changed = False
            for i in range(len(self.X)):
                self.mark[i] = False
                if np.abs(self._predict(self.X[i], 1, 1) - self.Y[i]) >= e:
                    if not changed:
                        changed = True
                else:
                    self.mark[i] = True
            if not changed:
                return
            q = qd
            qd = qualityR(self, Xt, Yt, sig)

    def predict(self, X, k, sig):
        return [self._predict(x, k, sig) for x in X]

    def _predict(self, x, k, sig):
        distances = []
        for i in range(len(self.X)):
            if not self.mark[i]:
                continue
            distances.append(euclidianDistance(x, self.X[i]))
        indices = np.argsort(distances)[:k]
        weights = [RBFkernel(distances[i], sig) for i in indices]
        w = sum(weights)
        nns = [self.Y[i] for i in indices]
        s = sum([nns[i] * weights[i] for i in range(len(indices))])
        prediction = s / w
        return prediction
    
    def numberOfExamples(self):
        return self.mark.sum()



#radial basis function kernel
def RBFkernel(distance, sig):
    return np.exp(-(distance * distance) / (2 * sig * sig))

# euclidian distance algorithm
def euclidianDistance(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.sum(diff * diff))

def quality(M, Xt, Yt):
    n = 0
    d = 0
    for x, y in zip(Xt, Yt):
        d += 1
        if M._predict(x, 1) == y:
            n += 1
    try:
        r = n / d
    except:
        r = 1
    return r


def qualityR(M, Xt, Yt, sig=10):
    e = 0
    for x, y in zip(Xt, Yt):
        ei = M._predict(x, 1, sig) - y
        e += np.abs(ei)
    return e / len(Xt)
