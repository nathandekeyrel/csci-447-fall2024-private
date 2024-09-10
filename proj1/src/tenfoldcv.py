import numpy as np
from sklearn.model_selection import KFold
from training import NaiveBayes
from evaluating import zero_one_loss, calculate_precision, calculate_recall, confusion_matrix


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

        cm = confusion_matrix(y_test, y_pred)
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

    print('')  # formatting

    return results


def summarize_results(results):
    avg_loss = np.mean([r["0/1 Loss"] for r in results])
    avg_precision = np.mean([np.mean(r["Precision"]) for r in results])
    avg_recall = np.mean([np.mean(r["Recall"]) for r in results])

    print("Overall Results:")
    print(f"Average 0/1 Loss: {avg_loss:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    print("\nConfusion Matrices:")
    for r in results:
        print(f"\nFold {r['Fold']}:")
        cm = r["Confusion Matrix"]
        if cm is not None and cm.size > 0:
            print(cm)
        else:
            print("Confusion matrix not available for this fold.")

    return avg_loss, avg_precision, avg_recall
