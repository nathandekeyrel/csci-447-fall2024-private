import numpy as np
from sklearn.model_selection import KFold
from training import NaiveBayes
from evaluating import zero_one_loss, calculate_precision, calculate_recall, confusion_matrix


def create_folds(X, y, n_splits=10, shuffle=True, random_state=44):
    """
    Create k-fold cross-validation splits of the data.

    :param X: numpy.ndarray, shape (n_samples, n_features)
        The input samples.
    :param y: numpy.ndarray, shape (n_samples,)
        The target values.
    :param n_splits: int, default=10
        Number of folds
    :param shuffle: boolean, default=True
        Whether to shuffle the data before splitting into batches.
    :param random_state: int
        Random seed for reproducibility.
    :return: list of tuples
        Each tuple contains (X_train, X_test, y_train, y_test) for a fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, X_test, y_train, y_test))

    return folds


def train_and_evaluate(X, y, num_classes):
    """
    Train and evaluate the Naive Bayes model using 10-fold cross-validation.

    This function performs the following steps:
    1. Create 10-fold splits of the data
    2. For each fold:
       - Train a Naive Bayes model
       - Make predictions on the test set
       - Calculate performance metrics
    3. Return the results for all folds

    :param num_classes: number of unique classes in the target column
    :param X: numpy.ndarray, shape (n_samples, n_features)
        The input samples.
    :param y: numpy.ndarray, shape (n_samples,)
        The target values.
    :return: list of dictionaries
        Each containing results for one fold
    """
    folds = create_folds(X, y)
    results = []
    summed_cm = np.zeros((num_classes, num_classes), dtype=int)

    for fold, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        model = NaiveBayes()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, num_classes)
        summed_cm += cm
        loss = zero_one_loss(y_test, y_pred)
        precision = calculate_precision(y_test, y_pred)
        recall = calculate_recall(y_test, y_pred)

        fold_results = {
            "Fold": fold,
            "0/1 Loss": loss,
            "Confusion Matrix": cm,
            "Precision": precision,
            "Recall": recall
        }
        results.append(fold_results)

    return results, summed_cm


def summarize_results(results, summed_cm):
    """
    Summarize and print the results from k-fold cross-validation.

    This function calculates and prints:
    - Average 0/1 Loss across all folds
    - Average Precision across all folds
    - Average Recall across all folds
    - Confusion Matrix for each fold

    :param summed_cm:
    :param results: list of dictionaries
        The results from train_and_evaluate function.
    :return: tuple (float, float, float)
        A tuple containing (avg_loss, avg_precision, avg_recall).
    """
    avg_loss = np.mean([r["0/1 Loss"] for r in results])
    avg_precision = np.mean([np.mean(r["Precision"]) for r in results])
    avg_recall = np.mean([np.mean(r["Recall"]) for r in results])

    print("Overall Results:")
    print(f"Average 0/1 Loss: {avg_loss:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    print("\nSummed Confusion Matrix:")
    print(summed_cm)

    print("-" * 50)

    return avg_loss, avg_precision, avg_recall
