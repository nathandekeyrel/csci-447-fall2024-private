import numpy as np
from sklearn.model_selection import KFold


def create_folds(X, y, n_splits=10, shuffle=True, random_state=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
