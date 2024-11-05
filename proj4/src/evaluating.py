import numpy as np
from sklearn.metrics import precision_score, recall_score


###################################
# Regression Loss Method
###################################

# Standard Means Squared Error
def mse(y_true, y_pred):
    """Calculate mean squared error for regression tasks

    :param y_true: array-like of shape (n_samples)
    :param y_pred: array-like, shape (n samples)
    :return: mean squared error
    """
    loss = np.mean((y_true - y_pred) ** 2)
    return loss


def rmse(y_true, y_pred):
    """Calculate root-mean squared error for regression tasks

    :param y_true: array-like of shape (n_samples)
    :param y_pred: array-like, shape (n samples)
    :return: root mean squared error
    """
    loss = np.sqrt(mse(y_true, y_pred))
    return loss


def mae(y_true, y_pred):
    """Calculate mean absolute error for regression tasks

    :param y_true: array-like of shape (n_samples)
    :param y_pred: array-like, shape (n samples)
    :return: mean absolute error
    """
    loss = np.mean(np.abs(y_true - y_pred))
    return loss


# I think we just go with the 3 different versions of mean squared error, some are more interpretable, some have
# more/less sensitivity to outliers, etc. I was reading that r^2 might not be great for NNs, a variation might
# work but from what I read these were the ones usually used.
# https://neptune.ai/blog/performance-metrics-in-machine-learning-complete-guide

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
    metric = np.mean(y_true != y_pred)
    return metric


# I think we just go with what we did for the first assignment: prec, recall, maybe a conf mat. If we decide to graph
# which might be beneficial it could be a bit better visually than just have a bunch of tables and conf mats.

# We can calculate these manually if you want, it's just easier to use sklearn in this instance
def calculate_precision(y_true, y_pred):
    """
    Calculate precision for each class.

    :param y_true: array-like of shape (n_samples)
        Same.
    :param y_pred: array-like of shape (n_samples)
        Same.
    :return: array-like of shape (n_classes)
        Precision for each class. If a class is not present in y_true or y_pred,
        the corresponding precision will be set to 0.
    """
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    return precision


def calculate_recall(y_true, y_pred):
    """
    Compute recall for each class.

    :param y_true: array-like of shape (n_samples)
        Same.
    :param y_pred: array-like of shape (n_samples)
        Same.
    :return: array-like of shape (n_classes)
        Recall for each class. If a class is not present in y_true,
        the corresponding recall will be set to 0.
    """
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    return recall
