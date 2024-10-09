import tenfoldcv as kfxv
import numpy as np
import knn
import editedKNN as eknn
import kMeans as km
import copy
import evaluating as ev
import preprocess as pr
import sys
from sklearn.metrics import r2_score

ks = [1, 3, 5, 7, 13, 15]
sigs = [0.25, 0.5, 1, 2]
es = [0.25, 0.5, 0.75]

def generateStartingTestData(X, Y):
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

def generateTrainingData(Xs, Ys, i):
    Xs = copy.copy(Xs)
    Ys = copy.copy(Ys)
    Xs.pop(i)
    Ys.pop(i)
    X_train = kfxv.mergedata(Xs)
    Y_train = kfxv.mergedata(Ys)
    return X_train, Y_train

def tuneKNNClassifier(X, Y):
    # generate the test sets and the folds
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = knn.KNNClassifier()
    perf = [0] * len(ks)
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        for j in range(len(ks)):
            predictions = cl.predict(X_test, ks[j])
            perf[j] += ev.zero_one_loss(Y_test, predictions)
    best = min(perf)
    index = perf.index(best)
    return ks[index]

def tuneKNNRegression(X, Y):
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = knn.KNNRegression()
    perf = [[0] * len(ks) for _ in range(len(sigs))]
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        for j in range(len(ks)):
            for k in range(len(sigs)):
                predictions = cl.predict(X_test, ks[j], sig=sigs[k])
                perf[k][j] += r2_score(Y_test, predictions)
    internal_arr_maxs = [max(perf[i]) for i in range(len(sigs))]
    max_r2 = max(internal_arr_maxs)
    sig_i = internal_arr_maxs.index(max_r2)
    k_i = perf[sig_i].index(max_r2)
    return ks[k_i], sigs[sig_i]

def tuneEKNNClassifier(X, Y):
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = eknn.EKNNErrClassifier()
    perf = [0] * len(ks)
    kc = sys.maxsize
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        cl.edit(X_test, Y_test)
        kc = min(kc, cl.numberOfExamples())
        for j in range(len(ks)):
            predictions = cl.predict(X_test, ks[j])
            perf[j] += ev.zero_one_loss(Y_test, predictions)
    best = min(perf)
    index = perf.index(best)
    return ks[index], kc

def tuneEKNNRegression(X, Y):
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = eknn.EKNNErrRegression()
    this_es = [es[i] * (np.max(Y) - np.min(Y)) for i in range(len(es))]
    perf = [[[0] * len(ks) for _ in range(len(sigs))] for _ in range(len(es))]
    kc = sys.maxsize
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        for j in range(len(es)):
            cl.edit(X_test, Y_test, 1, this_es[j])
            kc = min(kc, cl.numberOfExamples())
            for k in range(len(ks)):
                for l in range(len(sigs)):
                    predictions = cl.predict(X_test, ks[k], sigs[l])
                    perf[j][l][k] += r2_score(Y_test, predictions)
    max_val_mat = [[max(perf[i][j]) for i in range(len(es))] for j in range(len(sigs))]
    max_val_arr = [max(max_val_mat[i]) for i in range(len(sigs))]
    max_val = max(max_val_arr)
    sig_i = max_val_arr.index(max_val)
    e_i = max_val_mat[sig_i].index(max_val)
    k_i = perf[e_i][sig_i].index(max_val)
    cl.edit(X_test, Y_test, 1, this_es[e_i])
    return ks[k_i], sigs[sig_i], this_es[e_i], cl.numberOfExamples()

def tuneKMeansClassifier(X, Y, kc):
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    kmeans = km.KMeansClassification(kc)
    perf = [0] * len(ks)
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        kmeans.fit(X_train, Y_train)
        for j in range(len(ks)):
            predictions = kmeans.predict(X_test, ks[j])
            perf[j] += ev.zero_one_loss(Y_test, predictions)
    min_val = min(perf)
    k_i = perf.index(min_val)
    return ks[k_i]

def tuneKMeansRegression(X, Y, kc):
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    kmeans = km.KMeansRegression(kc)
    perf = [[0] * len(ks) for _ in range(len(sigs))]
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        kmeans.fit(X_train, Y_train)
        rX, rY = kmeans.get_reduced_dataset()
        cl = knn.KNNRegression()
        cl.fit(rX, rY)
        for j in range(len(sigs)):
            for k in range(len(ks)):
                predictions = cl.predict(X_test, ks[k], sigs[j])
                perf[j][k] += r2_score(Y_test, predictions)
    max_arr = [max(perf[i]) for i in range(len(sigs))]
    max_val = max(max_arr)
    sig_i = max_arr.index(max_val)
    k_i = perf[sig_i].index(max_val)
    return ks[k_i], sigs[sig_i]

def tuneEverything(datadirectory : str):
    if not datadirectory.endswith("/"):
        datadirectory += "/"
    filenames = ["soybean-small.data", "glass.data", "breast-cancer-wisconsin.data", "machine.data", "forestfires.csv", "abalone.data"]
    paths = [datadirectory + filenames[i] for i in range(6)]
    print("filename,k value for knn,k value for eknn,k cluster for kmeans,k value for kmeans\n")
    for i in range(3):
        path = paths[i]
        try:
            X, Y = pr.preprocess_data(path)
            print("Loading file at " + path + " successful")
        except:
            print("Failed to load file at " + path)
        kn = tuneKNNClassifier(X, Y)
        ken, kc = tuneEKNNClassifier(X, Y)
        kcn = tuneKMeansClassifier(X, Y, kc)
        print("%s,%d,%d,%d,%d\n" % (path, kn, ken, kc, kcn))
    print("filename,k value for knn,sig value for knn,k value for eknn,sig value for eknn,e value for eknn,k cluster for kmeans,k value for kmeans,sig value for kmeans\n")
    
    for i in range(3, 6):
        path = paths[i]
        try:
            X, Y = pr.preprocess_data(path)
            print("Loading file at " + path + " successful")
        except:
            print("Failed to load file at " + path)
        kn, sig_n = tuneKNNRegression(X, Y)
        ken, sig_e, e_e, kc = tuneEKNNRegression(X, Y)
        kcn, sig_cn = tuneKMeansRegression(X, Y, kc)
        print("%s,%d,%f,%d,%f,%d,%d,%d,%f\n" % (path, kn, sig_n, ken, sig_e, e_e, kc, kcn, sig_cn))