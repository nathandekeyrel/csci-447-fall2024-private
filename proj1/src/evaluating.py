import numpy as np
from sklearn import metrics


def zero_one_loss(y_true, y_pred):
    metric = np.mean(y_true != y_pred)
    return metric


# calc the confusion matrix
def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for i in range(len(y_true)):
        cm[np.where(classes == y_true[i])[0][0]][np.where(classes == y_pred[i])[0][0]] += 1
    return cm


def calculate_precision(y_true, y_pred):
    precision = metrics.precision_score(y_true, y_pred, average=None, zero_division=0)
    return precision


def calculate_recall(y_true, y_pred):
    recall = metrics.recall_score(y_true, y_pred, average=None, zero_division=0)
    return recall
