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


# RMSE

# MAE

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
