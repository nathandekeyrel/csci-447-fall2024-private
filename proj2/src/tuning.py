import tenfoldcv as kfxv
import numpy as np
import knn
import copy

NUMGUESSES = 10

def generateStuff(X, Y):
    # copy the data to prevent mutating the source
    X = copy.copy(X)
    Y = copy.copy(Y)
    # get the test set out of the data
    Xs, Ys = kfxv.kfold(X, Y, 10)
    X_test = np.array(Xs.pop(0))
    Y_test = np.array(Ys.pop(0))
    # generate the folds for tuning
    X = kfxv.mergedata(Xs)
    Y = kfxv.mergedata(Ys)
    Xs, Ys = kfxv.kfold(X, Y, 10)
    return Xs, Ys, X_test, Y_test

def tuneKNNClassifier(X, Y, mod = 0):
    # generate the test sets and the folds
    Xs, Ys, X_test, Y_test = generateStuff(X, Y)
    # get the number of samples
    sn = sum([len(Xs[i]) for i in range(len(Xs))])
    # get the square root of the number of samples and set that for the upper range
    high = int(np.sqrt(sn))
    # generate a list of k values to test
    ks = []
    for i in range(NUMGUESSES):
        ks.append(i + 1)
    ks = set(ks)
    ks = sorted(list(ks))
    cl = knn.KNNClassifier()
    perfs = []
    for i in range(10):
        Xsi = copy.copy(Xs)
        Ysi = copy.copy(Ys)
        Xsi.pop(i)
        Ysi.pop(i)
        X_train = kfxv.mergedata(Xsi)
        Y_train = kfxv.mergedata(Ysi)
        cl.fit(X_train, Y_train)
        perf = []
        for k in ks:
            predictions = cl.predict(X_test, k)
            perf.append([predictions[j] == Y_test[j] for j in range(len(Y_test))].count(True) / len(Y_test))
        perfs.append(perf)
    avgperfs = []
    for i in range(len(ks)):
        avgperfs.append(np.mean([perfs[j][i] for j in range(10)]))
    best = max(avgperfs)
    index = avgperfs.index(best)
    return ks[index]

def tuneKNNRegressor():
    pass

def tuneEKNNClassifier():
    pass

def tuneEKNNRegression():
    pass

def tuneKMeansClassifier():
    pass

def tuneKMeansRegression():
    pass