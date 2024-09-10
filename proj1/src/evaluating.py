import numpy as np
from sklearn import metrics


def zero_one_loss(y_true, y_pred):
    """
    Calculate the zero-one loss between true and predicted labels.

    :param y_true: array-like of shape (n_samples,)
        Ground truth (correct) labels.
    :param y_pred: array-like of shape (n_samples,)
        Predicted labels, returned by a classifier.
    :return: float
        The fraction of misclassifications (float between 0 and 1).
    """
    metric = np.mean(y_true != y_pred)
    return metric


# calc the confusion matrix
def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix to evaluate the accuracy of a classification.

    :param y_true: array-like of shape (n_samples,)
        Same as 0/1 loss.
    :param y_pred: array-like of shape (n_samples,)
        Same as 0/1 loss.
    :return: array, shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and predicted
        label being j-th class.
    """
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(y_true)):
        cm[np.where(classes == y_true[i])[0][0]][np.where(classes == y_pred[i])[0][0]] += 1
    return cm


def calculate_precision(y_true, y_pred):
    """
    Calculate precision for each class.

    :param y_true: array-like of shape (n_samples,)
        Same.
    :param y_pred: array-like of shape (n_samples,)
        Same.
    :return: array-like of shape (n_classes,)
        Precision for each class. If a class is not present in y_true or y_pred,
        the corresponding precision will be set to 0.
    """
    precision = metrics.precision_score(y_true, y_pred, average=None, zero_division=0)
    return precision


def calculate_recall(y_true, y_pred):
    """
    Compute recall for each class.

    :param y_true: array-like of shape (n_samples,)
        Same.
    :param y_pred: array-like of shape (n_samples,)
        Same.
    :return: array-like of shape (n_classes,)
        Recall for each class. If a class is not present in y_true,
        the corresponding recall will be set to 0.
    """
    recall = metrics.recall_score(y_true, y_pred, average=None, zero_division=0)
    return recall
