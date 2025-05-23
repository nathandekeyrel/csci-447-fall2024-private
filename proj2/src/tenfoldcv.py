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
    Vs.sort(key=lambda x: x[-1])
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


def _crossvalidationC(i, X, Y, nClasses, k, cl):
    """Perform cross-validation on a single fold for a given classifier

    :param i: Index of the fold to use as the holdout set
    :param X: List of feature vector folds
    :param Y: List of target value folds
    :param nClasses: Number of classes in the dataset
    :param k: Number of neighbors for kNN
    :param cl: Classifier object
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
    # fit the model to the training data
    cl.fit(Xt, Yt)
    # if there is an edit method, run that using the hold out set
    if hasattr(cl, 'edit') and callable(cl.edit):
        cl.edit(Xh, Yh)
    # get the predictions
    predictions = np.array(cl.predict(Xh, k))
    # return the actual values and the predictions
    return copy.copy(Yh), predictions


def tenfoldcrossvalidationC(cl, X, Y, k):
    """Perform 10-fold cross-validation for a classifier

    :param cl: Classifier object
    :param X: Feature vectors
    :param Y: Target values
    :param k: Number of neighbors for kNN
    :return: List of tuples (actual values, predictions) for each fold
    """
    nClasses = np.max(Y) + 1
    X, Y = kfold(X, Y, 10)
    results = [_crossvalidationC(i, X, Y, nClasses, k, cl) for i in range(10)]
    return results


def _crossvalidationR(i, X, Y, sig, k, re, e=0):
    """Perform cross-validation on a single fold for a given regression model

    :param i: Index of the fold to use as the holdout set
    :param X: List of feature vector folds
    :param Y: List of target value folds
    :param sig: Sigma value for the regression model
    :param k: Number of neighbors for kNN
    :param re: Regression model object
    :param e: Epsilon value for edited kNN (default 0)
    :return: Tuple of (actual values, predictions) for the holdout set
    """
    # copy the X and Y arrays to keep the sample data intact
    Xc = copy.copy(X)
    Yc = copy.copy(Y)
    # pop the ith element for the holdout set
    Xh = np.array(Xc.pop(i))
    Yh = np.array(Yc.pop(i))
    # merge the remaining data for the training set
    Xt = mergedata(Xc)
    Yt = mergedata(Yc)
    # fit the model to the training data
    re.fit(Xt, Yt)
    # if there is an edit method, run that using the hold out set
    if hasattr(re, 'edit') and callable(re.edit):
        re.edit(Xh, Yh, sig, e)
    # get the predictions
    predictions = np.array(re.predict(Xh, k, sig))
    # return a tuple containing the hold out target values and the predictions
    return copy.copy(Yh), predictions


def tenfoldcrossvalidationR(re, X, Y, k, sig, e=0):
    """Perform 10-fold cross-validation for a regression model

    :param re: Regression model object
    :param X: Feature vectors
    :param Y: Target values
    :param k: Number of neighbors for kNN
    :param sig: Sigma value for the regression model
    :param e: Epsilon value for edited kNN (default 0)
    :return: List of tuples (actual values, predictions) for each fold
    """
    X, Y = kfold(X, Y, 10)
    results = [_crossvalidationR(i, X, Y, sig, k, re, e) for i in range(10)]
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
