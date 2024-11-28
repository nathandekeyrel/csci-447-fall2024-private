import copy
import numpy as np
import random


def kfold(X, Y, k):
    """Stratify and break the data into k folds

    :param X: The feature vectors
    :param Y: The target values
    :param k: The number of folds
    :return: Xs, Ys (list[list], list[list])
        The folds of the feature vectors and their associated target values
    """
    # copy the vectors in D to a vector list Vs
    Vs = list(zip(X, Y))
    # shuffle the vectors in Vs
    random.shuffle(Vs)
    # sort the vectors in Vs by their class. Since the sort is stable, the randomization
    # introduced by the shuffle in the previous step will be preserved
    if isinstance(Y, np.ndarray):
        sortkey = lambda x: np.argmax(x[-1])
    else:
        sortkey = lambda x: x[-1]
    Vs.sort(key=sortkey)
    # initialize our list of vector lists Xs and Ys, which will represent the folds
    Xs = [[] for _ in range(k)]
    Ys = [[] for _ in range(k)]
    # iterate over the indices for each vector in Vs
    for i in range(len(Vs)):
        '''
        In this loop we use the clock property of modulus arithmetic to create stratification in our data.
        With this method we effectively map data like so:
          Given k = 3 and l = [0, 1, 2, 3, 4, 5]
          stratification(l) = [[0, 3], [1, 4], [2, 5]]
        '''
        Xs[i % k].append(Vs[i][0])
        Ys[i % k].append(Vs[i][1])
    # we return the list of lists produced by the instructions in this function
    return Xs, Ys


def _crossvalidation(i, X, Y, model):
    """Perform cross-validation on a single fold for a given classifier

    :param i: Index of the fold to use as the holdout set
    :param X: List of feature vector folds
    :param Y: List of target value folds
    :param model: The model being tested
    :return: Tuple of (actual values, predictions) for the holdout set
    """
    # copy the X and Y arrays to keep the sample data intact
    Xc = copy.copy(X)
    Yc = copy.copy(Y)
    # pop the ith element for the holdout set
    Xh = np.array(Xc.pop(i))
    Yh = np.array(Yc.pop(i))
    # merge the remaining data for the training set
    Xt = np.array(mergedata(Xc))
    Yt = np.array(mergedata(Yc))
    # initialize the classifier
    model.initialize_model(Xt, Yt)
    # train the model, using the holdout data to prevent overfitting
    model.train(Xh, Yh)
    # predict using the holdout data
    pred = model.predict(Xh)
    # return the actual values and the predictions
    return copy.copy(Yh), pred


def tenfoldcrossvalidation(X, Y, model):
    """Perform 10-fold cross-validation for a classifier

    :param X: Feature vectors
    :param Y: Target values
    :param model: The model being tested
    :return: List of tuples (actual values, predictions) for each fold
    """
    X, Y = kfold(X, Y, 10)
    results = [_crossvalidation(i, X, Y, model)
               for i in range(10)]
    return results


def mergedata(Ds):
    """Merge multiple data chunks into a single array

    :param Ds: List of data chunks
    :return: Merged data as a numpy array
    """
    # initialize our list that represents the merged data
    Dm = []
    # iterate through the chunks in Ds
    for D in Ds:
        # extend the merged data by the vectors in D
        Dm.extend(D)
    # return the vector list that resulted from the execution of this function
    return np.array(Dm)
