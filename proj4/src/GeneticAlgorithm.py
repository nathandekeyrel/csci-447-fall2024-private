import ffNN
import numpy as np
from copy import deepcopy as cp


class GeneticAlgorithm:
    def __init__(self, X_train, y_train, n_nodes_per_layer, n_hidden_layers, population, tournament_size,
                 is_classifier):
        """

        :param X_train: training features
        :param y_train: training target
        :param n_nodes_per_layer: number of nodes per hidden layer
        :param n_hidden_layers: number of hidden layers
        :param population: number of individual networks in the population
        :param tournament_size: number of individuals selected per tournament
        :param is_classifier: boolean, checks problem type
        """
        self.X_train = X_train
        self.y_train = y_train
        self.nets: list[ffNN.ffNN] = [None] * population
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.is_classifier = is_classifier
        self.best_perf = 0
        self.tournament_size = tournament_size
        self.best_network_index = 0
        self.initialize_model(X_train, y_train)

    def initialize_model(self, X_train, y_train):
        """Initialize the neural network models

        :param X_train: training features
        :param y_train: testing features
        """
        for i in range(len(self.nets)):
            self.nets[i] = ffNN.ffNN(X_train, y_train, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
            self.nets[i].weights = [np.random.uniform(low=-1.0, high=1.0, size=weight.shape) for weight in self.nets[i].weights]
            self.nets[i].biases = [np.random.uniform(low=-1.0, high=1.0, size=bias.shape) for bias in self.nets[i].biases]

    def _performance(self, y_pred, y_true):
        """Performance based on problem type.
                0/1 Loss for classification
                MSE for regression

        :param y_pred: predicted
        :param y_true: actual
        :return:
        """
        if self.is_classifier:
            results = np.mean(y_pred != y_true)
        else:
            results = np.mean(np.square(y_pred - y_true))
        return results

    def _tournament_selection(self, performances):
        """Given a random subset of the population, select the model with the highest performance

        :param performances:
        :return: network index of the winner
        """
        # random index of population
        tournament_indices = np.random.choice(len(performances), self.tournament_size, replace=False)
        tournament_performances = performances[tournament_indices]  # check each performance
        winner_tournament_index = np.argmin(tournament_performances)  # get minimum (smallest error)
        return tournament_indices[winner_tournament_index]

    def _arithmetic_crossover(self, parent1, parent2):
        """Follows the equation: (p1 + p2) / 2
        Calculated for each layer in the network

        :param parent1: the first tournament winner
        :param parent2: the second tournament winner
        :return: new neural network with calculated weights and biases
        """
        child_weights = [
            (parent1.weights[i] + parent2.weights[i]) / 2 for i in range(parent1.layers - 1)
        ]

        child_biases = [
            (parent1.biases[i] + parent2.biases[i]) / 2 for i in range(parent1.layers - 1)
        ]

        # initialize NN model
        child = ffNN.ffNN(self.X_train, self.y_train, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
        child.weights = child_weights  # give it the calced weights
        child.biases = child_biases  # give it the calced biases

        return child  # return the new child network

    @staticmethod
    def _gaussian_mutate(child, sigma=0.1):
        """Adds noise to the weights and biases of the child network

        :param child: The network created by crossover
        :param sigma: standard deviation
        :return: child network with mutated weights/biases
        """
        mutated_weights = [
            weights + np.random.normal(0, sigma, size=weights.shape) for weights in child.weights
        ]

        mutated_biases = [
            biases + np.random.normal(0, sigma, size=biases.shape) for biases in child.biases
        ]

        # give child the updated weights and biases
        child.weights = mutated_weights
        child.biases = mutated_biases

        return child

    def _train(self):
        """Helper method to train the model.
        """
        performances = np.array([
            self._performance(net.predict(self.X_train), self.y_train) for net in self.nets
        ])

        next_generation = []
        self.best_network_index = np.argmin(performances)

        next_generation.append(cp(self.nets[self.best_network_index]))

        while len(next_generation) < len(self.nets):
            # get the parents for crossover via tournament of size k
            parent1_index = self._tournament_selection(performances)
            parent2_index = self._tournament_selection(performances)

            # get child based on parents
            child = self._arithmetic_crossover(self.nets[parent1_index], self.nets[parent2_index])
            child = self._gaussian_mutate(child)  # add noise to child model

            next_generation.append(child)

        self.nets = next_generation

    def train(self, X_test, y_test):
        """Trains the actual network, implements early stopping and an improvement threshold
        in order to reduce the computational complexity of the model.

        :param X_test: testing feature vectors
        :param y_test: testing target vector
        :return: optimal model
        """
        best_perf = 0
        best_pop = None
        epochs = 0
        improvement_threshold = 0.001

        pred = self.predict(X_test)
        if self.is_classifier:
            best_perf = 1 - self._performance(pred, y_test)
        else:
            best_perf = 1 / self._performance(pred, y_test)
        best_pop = cp(self.nets)

        while epochs < 10:  # if no improvement after 10 iterations, return best model so far
            epochs += 1
            self._train()  # helper method to train models
            pred = self.predict(X_test)

            # performance based on problem type
            if self.is_classifier:
                perf = 1 - self._performance(pred, y_test)
            else:
                perf = 1 / self._performance(pred, y_test)

            # if performance increased and above the threshold, update best parameters
            if perf > best_perf and (perf - best_perf) > improvement_threshold:
                best_perf = perf
                best_pop = cp(self.nets)
                epochs = 0

        self.nets = cp(best_pop)

    def predict(self, X):
        """Given a set of feature vectors, predict the output

        :param X: feature vectors
        :return: prediction (target)
        """
        return self.nets[self.best_network_index].predict(X)
