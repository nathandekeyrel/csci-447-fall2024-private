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

# Global variables holding optimal k, sigma, and epsilon values
ks = [1, 3, 5, 7, 13, 15]
sigs = [0.25, 0.5, 1, 2]
es = [0.25, 0.5, 0.75]


def generateStartingTestData(X, Y):
    """Generate a starting test dataset from the input data.

    This function creates a test set and 10-fold cross-validation folds for tuning.

    :param X: The feature vector
    :param Y: The target vector
    :return: Tuple containing (Xs, Ys, X_test, Y_test)
             Xs, Ys are lists of 9 folds for cross-validation
             X_test, Y_test are the held-out test set
    """
    # copy the data to prevent mutating the source
    X = copy.copy(X)
    Y = copy.copy(Y)

    # get the test set out of the data (first fold)
    Xs, Ys = kfxv.kfold(X, Y, 10)
    X_test = np.array(Xs.pop(0))
    Y_test = np.array(Ys.pop(0))

    # generate the folds for tuning from remaining data
    X = kfxv.mergedata(Xs)
    Y = kfxv.mergedata(Ys)
    Xs, Ys = kfxv.kfold(X, Y, 10)
    return Xs, Ys, X_test, Y_test


def generateTrainingData(Xs, Ys, i):
    """Generate training data by excluding the i-th fold.

    :param Xs: List of feature vector folds
    :param Ys: List of target vector folds
    :param i: Index of the fold to exclude (used as validation set)
    :return: Tuple (X_train, Y_train) containing merged training data
    """
    Xs = copy.copy(Xs)
    Ys = copy.copy(Ys)
    Xs.pop(i)
    Ys.pop(i)
    X_train = kfxv.mergedata(Xs)
    Y_train = kfxv.mergedata(Ys)
    return X_train, Y_train


def tuneKNNClassifier(X, Y):
    """Tune the k parameter for KNN Classifier using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :return: Optimal k value
    """
    # generate the test sets and the folds
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = knn.KNNClassifier()
    perf = [0] * len(ks)

    # perform 10-fold cross-validation
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        for j in range(len(ks)):
            predictions = cl.predict(X_test, ks[j])
            perf[j] += ev.zero_one_loss(Y_test, predictions)

    # find the k value with the lowest error
    best = min(perf)
    index = perf.index(best)
    return ks[index]


def tuneKNNRegression(X, Y):
    """Tune the k and sigma parameters for KNN Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :return: Tuple (optimal k value, optimal sigma value)
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = knn.KNNRegression()
    perf = [[0] * len(ks) for _ in range(len(sigs))]

    # perform 10-fold cross-validation
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        for j in range(len(ks)):
            for k in range(len(sigs)):
                predictions = cl.predict(X_test, ks[j], sig=sigs[k])
                perf[k][j] += r2_score(Y_test, predictions)

    # find the k and sigma values with the highest R2 score
    internal_arr_maxs = [max(perf[i]) for i in range(len(sigs))]
    max_r2 = max(internal_arr_maxs)
    sig_i = internal_arr_maxs.index(max_r2)
    k_i = perf[sig_i].index(max_r2)
    return ks[k_i], sigs[sig_i]


def tuneEKNNClassifier(X, Y):
    """Tune the k parameter for Edited KNN Classifier using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :return: Tuple (optimal k value, number of examples after editing)
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = eknn.EKNNErrClassifier()
    perf = [0] * len(ks)
    kc = sys.maxsize  # initialize kc to maximum possible value

    # perform 10-fold cross-validation
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        cl.edit(X_test, Y_test)
        kc = min(kc, cl.numberOfExamples())  # update kc to the minimum number of examples after editing
        for j in range(len(ks)):
            predictions = cl.predict(X_test, ks[j])
            perf[j] += ev.zero_one_loss(Y_test, predictions)

    # find the k value with the lowest error
    best = min(perf)
    index = perf.index(best)
    return ks[index], kc


def tuneEKNNRegression(X, Y):
    """Tune the k, sigma, and epsilon parameters for Edited KNN Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :return: Tuple (optimal k value, optimal sigma value, optimal epsilon value, number of examples after editing)
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    cl = eknn.EKNNErrRegression()

    # scale epsilon values based on the range of Y
    this_es = [es[i] * (np.max(Y) - np.min(Y)) for i in range(len(es))]

    perf = [[[0] * len(ks) for _ in range(len(sigs))] for _ in range(len(es))]
    kc = sys.maxsize  # initialize kc to maximum possible value

    # perform 10-fold cross-validation
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        cl.fit(X_train, Y_train)
        for j in range(len(es)):
            cl.edit(X_test, Y_test, 1, this_es[j])
            kc = min(kc, cl.numberOfExamples())  # update kc to the minimum number of examples after editing
            for k in range(len(ks)):
                for l in range(len(sigs)):
                    predictions = cl.predict(X_test, ks[k], sigs[l])
                    perf[j][l][k] += r2_score(Y_test, predictions)

    # find the combination of parameters with the highest R2 score
    max_val_mat = [[max(perf[i][j]) for i in range(len(es))] for j in range(len(sigs))]
    max_val_arr = [max(max_val_mat[i]) for i in range(len(sigs))]
    max_val = max(max_val_arr)
    sig_i = max_val_arr.index(max_val)
    e_i = max_val_mat[sig_i].index(max_val)
    k_i = perf[e_i][sig_i].index(max_val)

    # perform final edit with optimal epsilon
    cl.edit(X_test, Y_test, 1, this_es[e_i])
    return ks[k_i], sigs[sig_i], this_es[e_i], cl.numberOfExamples()


def tuneKMeansClassifier(X, Y, kc):
    """Tune the k parameter for KMeans Classifier using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :param kc: Number of clusters for KMeans
    :return: Optimal k value for KNN prediction after KMeans clustering
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    kmeans = km.KMeansClassification(kc)
    perf = [0] * len(ks)

    # perform 10-fold cross-validation
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        kmeans.fit(X_train, Y_train)
        for j in range(len(ks)):
            predictions = kmeans.predict(X_test, ks[j])
            perf[j] += ev.zero_one_loss(Y_test, predictions)

    # find the k value with the lowest error
    min_val = min(perf)
    k_i = perf.index(min_val)
    return ks[k_i]


def tuneKMeansRegression(X, Y, kc):
    """Tune the k and sigma parameters for KMeans Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :param kc: Number of clusters for KMeans
    :return: Tuple (optimal k value, optimal sigma value) for KNN prediction after KMeans clustering
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    kmeans = km.KMeansRegression(kc)
    perf = [[0] * len(ks) for _ in range(len(sigs))]

    # perform 10-fold cross-validation
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

    # find the combination of parameters with the highest R2 score
    max_arr = [max(perf[i]) for i in range(len(sigs))]
    max_val = max(max_arr)
    sig_i = max_arr.index(max_val)
    k_i = perf[sig_i].index(max_val)
    return ks[k_i], sigs[sig_i]


def tuneEverything(datadirectory: str):
    """Tune parameters for all models (KNN, EKNN, KMeans) on multiple datasets.

    This function processes multiple datasets, tunes the parameters for classification
    and regression tasks, and prints the results.

    :param datadirectory: Directory containing the dataset files
    """
    if not datadirectory.endswith("/"):
        datadirectory += "/"
    filenames = ["soybean-small.data", "glass.data", "breast-cancer-wisconsin.data", "machine.data", "forestfires.csv",
                 "abalone.data"]
    paths = [datadirectory + filenames[i] for i in range(6)]

    # tune classification datasets
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

    # tune regression datasets
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
