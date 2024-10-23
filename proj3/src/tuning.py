import tenfoldcv as kfxv
import numpy as np
import ffNN as nn
import copy
import evaluating as ev
import preprocess as pr
import sys
from sklearn.metrics import r2_score

## TODO needs to be updated for current project

# global variables holding optimal k, sigma, and epsilon values
learning_rate = [1, 3, 5, 7, 13, 15]
batches = [0.25, 0.5, 1, 2]
n_hidden_layers = []


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

def tuneFFNNClassifier(X, Y):
    """Tune the k parameter for KNN Classifier using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :return: Optimal k value
    """
    # TODO hook in the classifier when it is done
    # generate the test sets and the folds
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    # cl = nn.ffNNClassification()
    perf = [0] * len(ks)

    # perform 10-fold cross-validation
    for i in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, i)
        # cl.fit(X_train, Y_train)
        for j in range(len(ks)):
            predictions = cl.predict(X_test)
            perf[j] += ev.zero_one_loss(Y_test, predictions)

    # find the k value with the lowest error
    
    return 


def tuneFFNNRegression(X, Y, n_hidden_layers):
    """Tune the k and sigma parameters for KNN Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :return: Tuple (optimal k value, optimal sigma value)
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    n_inputs = len(X[0])
    re = nn.ffNNRegression(n_inputs, n_hidden, n_hidden_layers)
    perfarr = np.array()

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
        # tune the classifier
        print("%s,%d,%d,%d,%d\n" % (path))

    # tune regression datasets
    print("filename,k value for knn,sig value for knn,k value for eknn,sig value for eknn,e value for eknn,k cluster for kmeans,k value for kmeans,sig value for kmeans\n")
    for i in range(3, 6):
        path = paths[i]
        try:
            X, Y = pr.preprocess_data(path)
            print("Loading file at " + path + " successful")
        except:
            print("Failed to load file at " + path)
        # tune the regressor
        print("%s,%d,%f,%d,%f,%d,%d,%d,%f\n" % (path))
