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
        qd = q
        while qd >= q:
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
                    Xe = np.insert(Xe, i, x, axis=0)
                    Ye = np.insert(Ye, i, y, axis=0)
                    i += 1
            if delta == 0:
                break
            qd = quality(self.cl, Xt, Yt)

    def predict(self, X, k):
        return [self._predict(x, k) for x in X]

    def _predict(self, x, k):
        return self.cl._predict(x, k)

class EKNNErrRegression:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def edit(self, Xt, Yt, sig, e):
        Xe = copy.copy(self.X)
        Ye = copy.copy(self.Y)
        self.cl = knn.KNNRegression()
        self.cl.fit(Xe, Ye)
        q = qualityR(self.cl, Xt, Yt, sig)
        qd = q
        print(str(qd) + " <= " + str(q))
        while qd <= q:
            delta = 0
            i = 0
            while i < len(Xe):
                x = Xe[i]
                y = Ye[i]
                Xe = np.delete(Xe, i, 0)
                Ye = np.delete(Ye, i, 0)
                self.cl.fit(Xe, Ye)
                p = self.cl._predict(x, 1, 1)
                pe = np.abs(p - y)
                if pe >= e:
                    print("tossed data")
                    delta += 1
                else:
                    print("kept data " + str(i) + " " + str(len(Xe)))
                    Xe = np.insert(Xe, i, x, axis=0)
                    Ye = np.insert(Ye, i, y, axis=0)
                    i += 1
            if delta == 0:
                break
            qd = qualityR(self.cl, Xt, Yt)
            print(str(qd) + " <= " + str(q))

    def predict(self, X, k, sig):
        return [self._predict(x, k, sig) for x in X]

    def _predict(self, x, k, sig):
        return self.cl._predict(x, k, sig)

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
