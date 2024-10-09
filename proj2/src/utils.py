import copy
import tenfoldcv as kfxv
import numpy as np


def generateTestData(X, Y):
    """Generates test data for doing cross validation

    :param X:
    :param Y:
    :return: X, Y, X_test, Y_test
    """
    # copy the data to prevent mutating the source
    X = copy.copy(X)
    Y = copy.copy(Y)
    # get the test set out of the data
    Xs, Ys = kfxv.kfold(X, Y, 10)
    X_test = np.array(Xs.pop(0))
    Y_test = np.array(Ys.pop(0))
    # generate the folds for tuning
    X = np.array(kfxv.mergedata(Xs))
    Y = np.array(kfxv.mergedata(Ys))
    return X, Y, X_test, Y_test
