import ffNN
import numpy as np
from copy import deepcopy as cp


class GeneticAlgorithm:
    def __init__(self, X_train, y_train, n_nodes_per_layer, n_hidden_layers, population, tournament_size,
                 is_classifier):
        self.X_train = X_train
        self.y_train = y_train
        self.nets: list[ffNN.ffNN] = [None] * population
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.is_classifier = is_classifier
        self.best_perf = 0
        self.tournament_size = tournament_size
        self.best_network_index = None
        self.initialize_model(X_train, y_train)

    def initialize_model(self, X_train, y_train):
        for i in range(len(self.nets)):
            self.nets[i] = ffNN.ffNN(X_train, y_train, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
            self.nets[i].weights = [np.random.uniform(low=-1.0, high=1.0, size=weight.shape) for weight in
                                    self.nets[i].weights]
            self.nets[i].biases = [np.random.uniform(low=-1.0, high=1.0, size=bias.shape) for bias in
                                   self.nets[i].biases]

    def _performance(self, y_pred, y_true):
        if self.is_classifier:
            results = np.mean(y_pred != y_true)
        else:
            results = np.mean(np.square(y_pred - y_true))
        return results

    def _tournament_selection(self, performances):
        tournament_indices = np.random.choice(len(performances), self.tournament_size, replace=False)
        tournament_performances = performances[tournament_indices]
        winner_tournament_index = np.argmin(tournament_performances)
        return tournament_indices[winner_tournament_index]

    def _arithmetic_crossover(self, parent1, parent2):
        child_weights = [
            (parent1.weights[i] + parent2.weights[i]) / 2 for i in range(parent1.layers - 1)
        ]

        child_biases = [
            (parent1.biases[i] + parent2.biases[i]) / 2 for i in range(parent1.layers - 1)
        ]

        child = ffNN.ffNN(self.X_train, self.y_train, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
        child.weights = child_weights
        child.biases = child_biases

        return child

    def _gaussian_mutate(self, child, sigma=0.1):
        mutated_weights = [
            weights + np.random.normal(0, sigma, size=weights.shape) for weights in child.weights
        ]

        mutated_biases = [
            biases + np.random.normal(0, sigma, size=biases.shape) for biases in child.biases
        ]

        child.weights = mutated_weights
        child.biases = mutated_biases

        return child

    def _train(self):
        performances = np.array([
            self._performance(net.predict(self.X_train), self.y_train) for net in self.nets
        ])

        next_generation = []
        self.best_network_index = np.argmin(performances)

        next_generation.append(cp(self.nets[self.best_network_index]))

        while len(next_generation) < len(self.nets):
            parent1_index = self._tournament_selection(performances)
            parent2_index = self._tournament_selection(performances)

            child = self._arithmetic_crossover(self.nets[parent1_index], self.nets[parent2_index])
            child = self._gaussian_mutate(child)

            next_generation.append(child)

        self.nets = next_generation

    def train(self, X_test, y_test):
        best_perf = 0
        best_pop = None
        epochs = 0
        while epochs < 25:
            epochs += 1
            self._train()
            pred = self.predict(X_test)

            if self.is_classifier:
                perf = 1 - self._performance(pred, y_test)
            else:
                perf = 1 / self._performance(pred, y_test)

            if perf > best_perf:
                best_perf = perf
                best_pop = cp(self.nets)
                epochs = 0

        self.nets = cp(best_pop)

    def predict(self, X):
        return self.nets[self.best_network_index].predict(X)
