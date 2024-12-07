import tenfoldcv as kfxv
import numpy as np
from ParticleSwarmOptimization import PSO
from DifferentialEvolution import DifferentialEvolution as DE
import copy

# ranges for the hyperparameters to be tuned
ps_range = [20, 60]
pso_inertia_range = [0, 1]
pso_cog_range = [0, 1]
pso_soc_range = [0, 1]
de_scaling_range = [0, 2]
de_binom_range = [0, 1]

# the number of tests to run
testsize = 25


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


def generateTrainingData(Xs, Ys, i):
    """Generate training data by excluding the i-th fold.

    :param Xs: List of feature vector folds
    :param Ys: List of target vector folds
    :param i: Index of the fold to exclude (used as validation set)
    :return: Tuple (X_train, Y_train) containing merged training data
    """
    Xs = copy.copy(Xs)
    Ys = copy.copy(Ys)
    Xs.pop(i)
    Ys.pop(i)
    X_train = kfxv.mergedata(Xs)
    Y_train = kfxv.mergedata(Ys)
    return X_train, Y_train


def tunePSO(X, Y, n_nodes_per_layer, n_hidden_layers, is_classifier):
    """Tune the k and sigma parameters for KNN Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :param n_nodes_per_layer: the number of nodes in each hidden layer
    :param n_hidden_layers: the number of hidden layers
    :param is_classifier: whether the model is for a classifier
    :return: the best performing population, inertia, cognitive weight, and social weight
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    pop_list = np.random.randint(ps_range[0], ps_range[1] + 1, testsize)
    inertia_list = np.random.rand(testsize) * (pso_inertia_range[1] - pso_inertia_range[0]) + pso_inertia_range[0]
    cog_list = np.random.rand(testsize) * (pso_cog_range[1] - pso_cog_range[0]) + pso_cog_range[0]
    soc_list = np.random.rand(testsize) * (pso_soc_range[1] - pso_soc_range[0]) + pso_soc_range[0]
    perfarr = np.zeros(testsize)
    # perform 10-fold cross-validation
    for n in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, n)
        for i in range(testsize):
            inertia = inertia_list[i]
            cog = cog_list[i]
            soc = soc_list[i]
            pop = pop_list[i]
            pso = PSO(X_train, Y_train, n_nodes_per_layer, n_hidden_layers, pop, inertia, cog, soc, is_classifier)
            pso.train(X_test, Y_test)
            perf = performance(pso, X_test, Y_test)
            perfarr[i] += perf
        print(n, "\n", perfarr)
    # get the indices and put them into a tuple
    index = np.argmin(perfarr)
    return pop_list[index], inertia_list[index], cog_list[index], soc_list[index]


def tuneDE(X, Y, n_nodes_per_layer, n_hidden_layers, is_classifier):
    """Tune the k and sigma parameters for KNN Regression using 10-fold cross-validation.

    :param X: Feature vector
    :param Y: Target vector
    :param n_nodes_per_layer: the number of nodes in each hidden layer
    :param n_hidden_layers: the number of hidden layers
    :param is_classifier: whether the model is for a classifier
    :return: the best performing population, scaling, and binomial crossover rate
    """
    Xs, Ys, X_test, Y_test = generateStartingTestData(X, Y)
    pop_list = np.random.randint(ps_range[0], ps_range[1] + 1, testsize)
    scaling_list = np.random.rand(testsize) * (de_scaling_range[1] - de_scaling_range[0]) + de_scaling_range[0]
    binom_list = np.random.rand(testsize) * (de_binom_range[1] - de_binom_range[0]) + de_binom_range[0]
    perfarr = np.zeros(testsize)
    # perform 10-fold cross-validation
    for n in range(10):
        X_train, Y_train = generateTrainingData(Xs, Ys, n)
        for i in range(testsize):
            scaling = scaling_list[i]
            binom = binom_list[i]
            pop = pop_list[i]
            de = DE(X_train, Y_train, n_nodes_per_layer, n_hidden_layers, pop, scaling, binom, is_classifier)
            de.train(X_test, Y_test)
            perf = performance(de, X_test, Y_test)
            perfarr[i] += perf
        print(n, "\n", perfarr)
    # get the indices and put them into a tuple
    index = np.argmin(perfarr)
    return pop_list[index], scaling_list[index], binom_list[index]


def performance(model, X_test, Y_test):
    """Helper function for finding the performance of a model. Inexplicably different from the other ones
    
    :param model: the model that is being tested
    :param X_test: the X vectors to be used to judge performance
    :param Y_test: the target values
    """
    pred = model.predict(X_test)
    if model.is_classifier:
        results = np.mean(pred != Y_test)
    else:
        results = np.mean(np.square(pred - Y_test))
    return results
