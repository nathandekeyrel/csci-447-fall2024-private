import numpy as np
from sklearn.metrics import mean_squared_error


###################################
# Regression Loss Method
###################################

def mse(y_true, y_pred):
    """Calculate mean squared error for regression tasks

    :param y_true: array-like of shape (n_samples)
    :param y_pred: array-like, shape (n samples)
    :return:
    """
    return mean_squared_error(y_true, y_pred)


###################################
# Classification Loss Methods
###################################

def zero_one_loss(y_true, y_pred):
    """Calculate the zero-one loss between true and predicted labels.

    :param y_true: array-like of shape (n_samples)
        Ground truth (correct) labels.
    :param y_pred: array-like of shape (n_samples)
        Predicted labels, returned by a classifier.
    :return: float
        The fraction of misclassifications (float between 0 and 1).
    """
    pre = (y_true != y_pred)
    metric = np.mean(y_true != y_pred)
    return metric
