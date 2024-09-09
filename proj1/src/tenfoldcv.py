import numpy as np
from sklearn.model_selection import KFold
from training import NaiveBayes
from evaluating import zero_one_loss, create_confusion_matrix, calculate_precision, calculate_recall


def create_folds(X, y, n_splits=10, shuffle=True, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, X_test, y_train, y_test))

    return folds


def train_and_evaluate(X, y):
    folds = create_folds(X, y)
    results = []

    for fold, (X_train, X_test, y_train, y_test) in enumerate(folds, 1):
        model = NaiveBayes()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = zero_one_loss(y_test, y_pred)
        conf_mat = create_confusion_matrix(y_test, y_pred)
        precision = calculate_precision(y_test, y_pred)
        recall = calculate_recall(y_test, y_pred)

        fold_results = {
            "Fold": fold,
            "0/1 Loss": loss,
            "Confusion Matrix": conf_mat,
            "Precision": precision,
            "Recall": recall
        }
        results.append(fold_results)

        print(f"Fold {fold} completed")

    return results


def summarize_results(results):
    avg_loss = np.mean([r["0/1 Loss"] for r in results])
    avg_precision = np.mean([r["Precision"] for r in results])
    avg_recall = np.mean([r["Recall"] for r in results])

    print("\nOverall Results:")
