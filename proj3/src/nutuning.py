import tenfoldcv as kfxv
import numpy as np
import ffNN as nn
import copy
import evaluating as ev
import preprocess as pr
import sys
from sklearn.metrics import r2_score

#ranges for the hyperparameters to be tuned
learning_rate = (-2, 2)
batches = (0, 1.0)
n_hidden = (0.5, 2)
momentum = (0, 1.0)

#the number of tests to run
testsize = 100

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


def tuneFFNNRegression(X, Y, n_hidden_layers):
    """Tune the k and sigma parameters for KNN Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :param n_hidden_layers: the number of hidden layers
    :return learning_rate:
    :return batch_size: batch size as a factor of number of samples
    :return n_hidden: the number of hidden nodes per layer as a factor of number of inputs
    :return momentum: 
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    n_inputs = len(X[0])
    m = np.max(X)
    lrlist = np.power(2, (np.random.rand(testsize) * (learning_rate[1] - learning_rate[0]) - learning_rate[1] - np.log2(m)))
    batchlist = (np.random.rand(testsize) * np.floor(len(X) * 0.81) + 1).astype(int)
    nhnlist = ((np.random.rand(testsize) * (n_hidden[1] - n_hidden[0]) + n_hidden[0]) * len(X[0])).astype(int)
    momlist = np.random.rand(testsize)
    perfarr = np.zeros(testsize)
    # perform 10-fold cross-validation
    for n in range(10):
        print("Tuning set", n)
        X_train, Y_train = generateTrainingData(Xs, Ys, n)
        n_X = len(X_train)
        for i in range(testsize):
            learning       = lrlist[i]
            batch_size     = min(n_X, max(1, batchlist[i]))
            n_hidden_nodes = nhnlist[i]
            mom            = momlist[i]
            re = nn.ffNNRegression(X_train, Y_train, n_inputs, n_hidden_nodes, n_hidden_layers)
            # We train the model one epoch at a time until it stops improving
            # bestresults = np.square(np.max(Y_test) - np.min(Y_test))
            bestresults = 0
            ephochs_since_last_improvement = 0
            epochs = 0
            while ephochs_since_last_improvement < 10: # if it doesn't improve after 20 iterations it ends
                epochs += 1
                ephochs_since_last_improvement += 1
                re.train(1, batch_size, learning, mom)
                predictions = re.predict(X_test)
                results = 1 / ev.mse(Y_test, predictions)
                if results > bestresults:
                    ephochs_since_last_improvement = 0
                    bestresults = results
            perfarr[i] += bestresults
        print(perfarr)
    # get the idices and put them into a tuple. I got this off of chatgpt because I'm not spending 2 hours looking for the specific algorithm that does this
    index = np.argmax(perfarr)

    return lrlist[index], batchlist[index], nhnlist[index], momlist[index]


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
