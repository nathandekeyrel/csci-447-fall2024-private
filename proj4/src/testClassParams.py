import numpy as np
import copy
from preprocess import preprocess_data
import tenfoldcv as kfxv
import ffNN as nn
import evaluating as ev


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


def test_parameter(X, Y, param_name, param_values, base_params):
    """Test individual parameters and hold others constant. Faster to get ranges than running whole thing.

    :param X: The feature vector
    :param Y: The target vector
    :param param_name: the parameter being tested
    :param param_values: values within parameter to be tested
    :param base_params: other params held constant
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    n_input = X.shape[1]
    n_output = len(np.unique(Y))

    print(f"\nTesting different {param_name} values...")
    print(f"Base parameters held constant:")
    for key, value in base_params.items():
        if key != param_name:
            print(f"  {key}: {value}")
    print()

    results = {}
    for param_value in param_values:
        total_loss = 0
        print(f"Testing {param_name} = {param_value}")

        for i in range(5):
            X_val = np.array(Xs[i])
            Y_val = np.array(Ys[i])
            X_train, Y_train = get_train_data(Xs, Ys, i)

            # update parameter being tested
            current_params = base_params.copy()
            current_params[param_name] = param_value

            model = nn.ffNNClassification(
                n_input=n_input,
                n_hidden=current_params['n_nodes'],
                n_hidden_layers=current_params['n_hidden_layers'],
                n_output=n_output
            )

            model.train(
                X=X_train,
                y=Y_train,
                epochs=current_params['epochs'],
                batchsize=current_params['batch_size'],
                learning_rate=current_params['learning_rate'],
                momentum=current_params['momentum']
            )

            y_pred = model.predict(X_val)
            fold_loss = ev.zero_one_loss(Y_val, y_pred)
            total_loss += fold_loss
            print(f"    Fold {i + 1}: loss = {fold_loss:.4f}")

        avg_loss = total_loss / 5
        results[param_value] = avg_loss
        print(f"{param_name}: {param_value}, Average loss: {avg_loss:.4f}\n")

    return results


def test_all_parameters(X, Y):
    """Test each parameter individually while holding others constant"""

    # base parameters
    base_params = {
        'n_nodes': 20,
        'learning_rate': 0.5,
        'momentum': 0.9,
        'batch_size': 128,
        'epochs': 50,
        'n_hidden_layers': 1
    }

    # test diff numbers of hidden nodes
    node_values = [1, 5, 10, 15, 20, 25, 30, 40, 50, 100, 150]
    node_results = test_parameter(X, Y, 'n_nodes', node_values, base_params)

    # test diff learning rates
    lr_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    lr_results = test_parameter(X, Y, 'learning_rate', lr_values, base_params)

    # test diff momentum values
    momentum_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95]
    momentum_results = test_parameter(X, Y, 'momentum', momentum_values, base_params)

    # test diff batch sizes
    batch_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    batch_results = test_parameter(X, Y, 'batch_size', batch_values, base_params)

    # test diff epoch values
    epoch_values = [1, 5, 10, 20, 50, 75, 100, 150, 200]
    epoch_results = test_parameter(X, Y, 'epochs', epoch_values, base_params)

    # test diff numbers of hidden layers
    layer_values = [0, 1, 2]
    layer_results = test_parameter(X, Y, 'n_hidden_layers', layer_values, base_params)

    # summary
    print("\nSummary of Results:")
    print("\nHidden Nodes:")
    for nodes, loss in node_results.items():
        print(f"  {nodes}: {loss:.4f}")

    print("\nLearning Rates:")
    for lr, loss in lr_results.items():
        print(f"  {lr}: {loss:.4f}")

    print("\nMomentum Values:")
    for m, loss in momentum_results.items():
        print(f"  {m}: {loss:.4f}")

    print("\nBatch Sizes:")
    for bs, loss in batch_results.items():
        print(f"  {bs}: {loss:.4f}")

    print("\nEpoch Values:")
    for e, loss in epoch_results.items():
        print(f"  {e}: {loss:.4f}")

    print("\nHidden Layers:")
    for l, loss in layer_results.items():
        print(f"  {l}: {loss:.4f}")


if __name__ == "__main__":
    print("Param Testing for Classification...")

    # load+preprocess
    cancer_filepath = "../data/breast-cancer-wisconsin.data"
    X, Y = preprocess_data(cancer_filepath)

    choice = ''
    while choice != '8':
        # menu to select which parameter to test
        print("\nWhat would you like to test?")
        print("1. Hidden Nodes")
        print("2. Learning Rate")
        print("3. Momentum")
        print("4. Batch Size")
        print("5. Epochs")
        print("6. Number of Hidden Layers")
        print("7. All Parameters")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        # base parameters
        base_params = {
            'n_nodes': 20,
            'learning_rate': 0.5,
            'momentum': 0.9,
            'batch_size': 128,
            'epochs': 50,
            'n_hidden_layers': 1
        }

        if choice == '1':
            node_values = [1, 5, 10, 15, 20, 25, 30, 40, 50, 100, 150]
            test_parameter(X, Y, 'n_nodes', node_values, base_params)
        elif choice == '2':
            lr_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
            test_parameter(X, Y, 'learning_rate', lr_values, base_params)
        elif choice == '3':
            momentum_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95]
            test_parameter(X, Y, 'momentum', momentum_values, base_params)
        elif choice == '4':
            batch_values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
            test_parameter(X, Y, 'batch_size', batch_values, base_params)
        elif choice == '5':
            epoch_values = [1, 5, 10, 20, 50, 75, 100, 150, 200]
            test_parameter(X, Y, 'epochs', epoch_values, base_params)
        elif choice == '6':
            layer_values = [0, 1, 2]
            test_parameter(X, Y, 'n_hidden_layers', layer_values, base_params)
        elif choice == '7':
            test_all_parameters(X, Y)
        elif choice == '8':
            print("byebyeyebyey")
            exit(0)
