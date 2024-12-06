import tenfoldcv as kfxv
import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
import copy
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


def tune_ga_params(X, Y, n_nodes_per_layer, n_hidden_layers, is_classifier):
    """Tune GA hyperparameters using 10-fold cross-validation.

    :param X: Feature vectors
    :param Y: Target values
    :param n_nodes_per_layer: Number of nodes per hidden layer
    :param n_hidden_layers: Number of hidden layers
    :param is_classifier: Boolean indicating if this is a classification task
    :return: Tuple of (best_params, results_dict)
    """
    tournament_sizes = [2, 3, 4]  # smaller k due to higher selection pressure
    population_sizes = [30, 50, 80]

    Xs, Ys = kfxv.kfold(X, Y, 10)

    best_perf = 0
    best_params = None
    results = {}

    for k in tournament_sizes:
        for pop in population_sizes:
            total_perf = 0

            for i in range(10):
                # validation folds
                X_val = np.array(Xs[i])
                Y_val = np.array(Ys[i])

                # merge remaining folds for training
                X_train, Y_train = kfxv.mergedata([Xs[j] for j in range(10) if j != i]), \
                    kfxv.mergedata([Ys[j] for j in range(10) if j != i])

                # initialize ga model
                ga = GeneticAlgorithm(
                    X_train, Y_train,
                    n_nodes_per_layer=n_nodes_per_layer,
                    n_hidden_layers=n_hidden_layers,
                    population=pop,
                    tournament_size=k,
                    is_classifier=is_classifier
                )

                ga.train(X_val, Y_val)

                # performance, yeah I know it accesses private methods but i don't think it matters
                y_pred = ga.predict(X_val)
                perf = 1 - ga._performance(y_pred, Y_val) if is_classifier else \
                    1 / ga._performance(y_pred, Y_val)

                total_perf += perf

            avg_perf = total_perf / 10
            results[(k, pop)] = avg_perf

            print(f"k={k}, pop={pop}: avg_performance={avg_perf:.4f}")

            if avg_perf > best_perf:
                best_perf = avg_perf
                best_params = (k, pop)
                print(f"New best parameters found! Performance: {avg_perf:.4f}")

    return best_params, results


if __name__ == "__main__":
    ###################################
    # Tune Classification
    ###################################

    # # load and preprocess classification datasets
    # cancer_filepath = "../data/breast-cancer-wisconsin.data"
    # glass_filepath = "../data/glass.data"
    # soybean_filepath = "../data/soybean-small.data"
    #
    # X_cancer, y_cancer = preprocess_data(cancer_filepath)
    # X_glass, y_glass = preprocess_data(glass_filepath)
    # X_soybean, y_soybean = preprocess_data(soybean_filepath)
    #
    # classification_configs = {
    #     'Cancer': {
    #         'X': X_cancer,
    #         'y': y_cancer,
    #         'nodes': 5,
    #     },
    #     'Glass': {
    #         'X': X_glass,
    #         'y': y_glass,
    #         'nodes': 7,
    #     },
    #     'Soybean': {
    #         'X': X_soybean,
    #         'y': y_soybean,
    #         'nodes': 38,
    #     }
    # }
    #
    # print("\nClassification Results:")
    # for name, config in classification_configs.items():
    #     print(f"\n{name} Dataset:")
    #     for n_hidden in [0, 1, 2]:
    #         print(f"\n{n_hidden} hidden layers:")
    #         best_params, results = tune_ga_params(
    #             config['X'], config['y'],
    #             n_nodes_per_layer=config['nodes'],
    #             n_hidden_layers=n_hidden,
    #             is_classifier=True
    #         )
    #         print(f"Best parameters found:")
    #         print(f"Tournament size: {best_params[0]}")
    #         print(f"Population size: {best_params[1]}")
    #         print(f"Performance: {results[best_params]:.4f}")

    ###################################
    # Tune Regression
    ###################################

    # abalone_filepath = "../data/abalone.data"
    hardware_filepath = "../data/machine.data"
    fires_filepath = "../data/forestfires.csv"

    # X_abalone, y_abalone = preprocess_data(abalone_filepath)
    X_hardware, y_hardware = preprocess_data(hardware_filepath)
    X_fires, y_fires = preprocess_data(fires_filepath)

    regression_configs = {
        # 'Abalone': {
        #     'X': X_abalone,
        #     'y': y_abalone,
        #     'nodes': {
        #         0: 13,
        #         1: 13,
        #         2: 12
        #     }
        # },
        'Hardware': {
            'X': X_hardware,
            'y': y_hardware,
            'nodes': {
                0: 30,
                1: 28,
                2: 64
            }
        },
        'Fires': {
            'X': X_fires,
            'y': y_fires,
            'nodes': {
                0: 54,
                1: 25,
                2: 34
            }
        }
    }

    # Run regression tuning
    print("\nRegression Results:")
    for name, config in regression_configs.items():
        print(f"\n{name} Dataset:")
        for n_hidden in [0, 1, 2]:
            print(f"\n{n_hidden} hidden layers:")
            best_params, results = tune_ga_params(
                config['X'], config['y'],
                n_nodes_per_layer=config['nodes'][n_hidden],
                n_hidden_layers=n_hidden,
                is_classifier=False
            )
            print(f"Best parameters found:")
            print(f"Tournament size: {best_params[0]}")
            print(f"Population size: {best_params[1]}")
            print(f"Performance: {results[best_params]:.4f}")
