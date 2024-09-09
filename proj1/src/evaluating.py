import numpy as np
from sklearn import metrics


def zero_one_loss(y_true, y_pred):
    metric = np.mean(y_true != y_pred)
    return metric


def create_confusion_matrix(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    return cm


def calculate_precision(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average=None)
    return precision


def calculate_recall(y_true, y_pred):
    recall = metrics.recall_score(y_true, y_pred, average=None)
    return recall
