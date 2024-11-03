import tenfoldcv as kfxv
import numpy as np
import ffNN as nn
import copy
import evaluating as ev
from preprocess import preprocess_data


def generateStartingTestData(X, Y):
    """Generate a starting test dataset from the input data.

    This function creates a test set and 10-fold cross-validation folds for tuning.

    :param X: The feature vector
    :param Y: The target vector
    :return: Tuple containing (Xs, Ys, X_test, Y_test)
             Xs, Ys are lists of 9 folds for cross-validation
             X_test, Y_test are the held-out test set
    """
    # copy the data to prevent mutating the source
    X = copy.copy(X)
    Y = copy.copy(Y)

    # get the test set out of the data (first fold)
    Xs, Ys = kfxv.kfold(X, Y, 10)
    X_test = np.array(Xs.pop(0))
    Y_test = np.array(Ys.pop(0))

    # generate the folds for tuning from remaining data
    X = kfxv.mergedata(Xs)
    Y = kfxv.mergedata(Ys)
    Xs, Ys = kfxv.kfold(X, Y, 10)
    return Xs, Ys, X_test, Y_test


def get_train_data(Xs, Ys, i):
    """Helper function to merge training data excluding validation fold.

    :param Xs: List of feature vector folds
    :param Ys: List of target vector folds
    :param i: Index of the fold to exclude (used as validation set)
    :return: Tuple (X_train, Y_train) containing merged training data as numpy arrays
    """
    X_train = []
    Y_train = []
    for j in range(len(Xs)):
        if j != i:  # skip validation fold
            X_train.extend(Xs[j])
            Y_train.extend(Ys[j])
    return np.array(X_train), np.array(Y_train)


def get_hidden_nodes(n_input, n_output):
    """Calculate hidden nodes as a factor of input/output sizes

    :param n_input: Number of input nodes in neural network
    :param n_output: Number of output nodes in neural network
    :return: List containing single value - number of hidden nodes to use
    """
    return [(n_input + n_output) // 2]


def tuneNNClassifier(X, Y, n_hidden_layers):
    """Tune neural network using fixed parameter ranges for classification tasks.
    Preforms grid search over learning rates, momentum, and batch sizes. Implements
    early stopping with patience to prevent overfitting.

    :param X: features from dataset
    :param Y: target vector of class labels
    :param n_hidden_layers: number of hidden layers in the network architecture
    :return: dictionary containing the best parameters found:
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    n_input = X.shape[1]
    n_output = len(np.unique(Y))

    # ranges for grid search
    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    momentum_values = [0.7, 0.8, 0.9, 0.95]
    batch_sizes = [4, 8, 16, 32, 64, 128]
    hidden_nodes = get_hidden_nodes(n_input, n_output)[0]

    best_params = {
        'hidden_nodes': hidden_nodes,
        'learning_rate': 0,
        'momentum': 0,
        'batch_size': 0,
        'n_hidden_layers': n_hidden_layers,
        'best_score': float('inf')
    }

    for lr in learning_rates:
        for momentum in momentum_values:
            for batch_size in batch_sizes:
                total_loss = 0

                for i in range(10):  # ten-fold cv
                    X_val = np.array(Xs[i])  # current fold is validation set
                    Y_val = np.array(Ys[i])
                    X_train, Y_train = get_train_data(Xs, Ys, i)

                    model = nn.ffNNClassification(
                        n_input=n_input,
                        n_hidden=hidden_nodes,
                        n_hidden_layers=n_hidden_layers,
                        n_output=n_output
                    )

                    # early stopping stuff
                    best_fold_loss = float('inf')
                    epochs_since_improvement = 0
                    epoch = 0
                    min_epochs = 20  # force training for x epochs
                    patience = 10  # stop after x epochs without improvement

                    while (epochs_since_improvement < patience) or (epoch < min_epochs):
                        epoch += 1
                        epochs_since_improvement += 1

                        model.train(
                            X=X_train,
                            y=Y_train,
                            epochs=1,
                            batchsize=batch_size,
                            learning_rate=lr,
                            momentum=momentum
                        )

                        if epoch % 5 == 0 or epoch < min_epochs:  # evaluate every 5 epochs
                            y_pred = model.predict(X_val)
                            fold_loss = ev.zero_one_loss(Y_val, y_pred)

                            if fold_loss < best_fold_loss:
                                best_fold_loss = fold_loss
                                epochs_since_improvement = 0

                    total_loss += best_fold_loss

                avg_loss = total_loss / 10

                print(f"LR: {lr:.2f}, Momentum: {momentum:.2f}, Batch: {batch_size}, Loss: {avg_loss:.4f}")

                # update with current best scores
                if avg_loss < best_params['best_score']:
                    best_params.update({
                        'hidden_nodes': hidden_nodes,
                        'learning_rate': lr,
                        'momentum': momentum,
                        'batch_size': batch_size,
                        'best_score': avg_loss
                    })
                    print(f"New best score: {avg_loss:.4f}")

    return best_params


if __name__ == "__main__":

    ###################################
    # Tune Classification
    ###################################

    # load datasets
    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    glass_filepath = "../data/glass.data"
    soybean_filepath = "../data/soybean-small.data"

    # preprocess data
    X_cancer, y_cancer = preprocess_data(cancer_filepath)
    X_glass, y_glass = preprocess_data(glass_filepath)
    X_soybean, y_soybean = preprocess_data(soybean_filepath)

    # print results for Cancer dataset
    print("\nCancer Results:")
    for n_hidden in [0, 1, 2]:
        print(f"\n{n_hidden} hidden layers:")
        cancer_results = tuneNNClassifier(X_cancer, y_cancer, n_hidden)
        print(f"Learning Rate: {cancer_results['learning_rate']:.4f}")
        print(f"Momentum: {cancer_results['momentum']:.4f}")
        print(f"Batch Size: {cancer_results['batch_size']}")
        print(f"Hidden Nodes: {cancer_results['hidden_nodes']}")
        print(f"Best Score: {cancer_results['best_score']:.4f}")

    # print results for Glass dataset
    print("\nGlass Results:")
    for n_hidden in [0, 1, 2]:
        print(f"\n{n_hidden} hidden layers:")
        glass_results = tuneNNClassifier(X_glass, y_glass, n_hidden)
        print(f"Learning Rate: {glass_results['learning_rate']:.4f}")
        print(f"Momentum: {glass_results['momentum']:.4f}")
        print(f"Batch Size: {glass_results['batch_size']}")
        print(f"Hidden Nodes: {glass_results['hidden_nodes']}")
        print(f"Best Score: {glass_results['best_score']:.4f}")

    # print results for Soybean dataset
    print("\nSoybean Results:")
    for n_hidden in [0, 1, 2]:
        print(f"\n{n_hidden} hidden layers:")
        soybean_results = tuneNNClassifier(X_soybean, y_soybean, n_hidden)
        print(f"Learning Rate: {soybean_results['learning_rate']:.4f}")
        print(f"Momentum: {soybean_results['momentum']:.4f}")
        print(f"Batch Size: {soybean_results['batch_size']}")
        print(f"Hidden Nodes: {soybean_results['hidden_nodes']}")
        print(f"Best Score: {soybean_results['best_score']:.4f}")
