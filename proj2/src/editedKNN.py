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

    def edit(self, Xt, Yt):
        Xe = copy.copy(self.X)
        Ye = copy.copy(self.Y)
        self.cl = knn.KNNClassifier()
        self.cl.fit(Xe, Ye)
        q = quality(self.cl, Xt, Yt)
        while quality(self.cl, Xt, Yt) >= q:
            delta = 0
            i = 0
            while i < len(Xe):
                x = Xe[i]
                y = Ye[i]
                Xe = np.delete(Xe, i, 0)
                Ye = np.delete(Ye, i, 0)
                self.cl.fit(Xe, Ye)
                p = self.cl._predict(x, 1)
                if not p == y:
                    delta += 1
                else:
                    np.insert(Xe, i, x, axis=0)
                    np.insert(Ye, i, y, axis=0)
                    i += 1
            if delta > 0:
                break

    def predict(self, X, k):
        return [self._predict(x, k) for x in X]

    def _predict(self, x, k):
        return self.cl._predict(x, k)


def quality(C, Xt, Yt):
    n = 0
    d = 0
    for x, y in zip(Xt, Yt):
        d += 1
        if C.predict(x, 1) == y:
            n += 1
    try:
        r = n / d
    except:
        r = 1
    return r


def qualityR(R, T, k):
    e = 0
    for t in T:
        te = R.predict(t, k)
        e += (te * te)
    return e / len(T)
