import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def zero_one_loss(y_true, y_pred):
    """
    Calculate the zero-one loss between true and predicted labels.

    :param y_true: array-like of shape (n_samples)
        Ground truth (correct) labels.
    :param y_pred: array-like of shape (n_samples)
        Predicted labels, returned by a classifier.
    :return: float
        The fraction of misclassifications (float between 0 and 1).
    """
    metric = np.mean(y_true != y_pred)
    return metric


def confusion_matrix(y_true, y_pred, num_classes, class_names):
    """
    Calculate confusion matrix to evaluate the accuracy of a classification.

    :param class_names: unique class names
    :param y_true: array-like of shape (n_samples)
        Ground truth (correct) target values.
    :param y_pred: array-like of shape (n_samples)
        Estimated targets as returned by a classifier.
    :param num_classes: int
        Number of unique classes in the target column.
    :return: array, shape (num_classes, num_classes)
        Confusion matrix whose i-th row and j-th column entry indicates the
        number of samples with true label being i-th class and predicted
        label being j-th class.
    """
    classes = np.unique(class_names)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        i, j = class_to_index[t], class_to_index[p]
        cm[i, j] += 1

    return cm


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


def calculate_f_score(y_true, y_pred):
    """
    Compute f-score for each class
    :param y_true: array-like of shape (n_samples)
    :param y_pred: array-like of shape (n_samples)
    :return: array-like of shape (n_samples)
        F1 score for each class.
    """
    f_score = f1_score(y_true, y_pred, labels=None, average=None)
    return f_score
