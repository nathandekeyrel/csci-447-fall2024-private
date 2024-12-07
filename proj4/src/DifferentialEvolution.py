import ffNN
import numpy as np
from copy import deepcopy as cp
import threading as th
import multiprocessing as mp


class DifferentialEvolution:
    def __init__(self, X_train, y_train, n_nodes_per_layer, n_hidden_layers, population, scaling, binomial_crossover,
                 is_classifier):
        """initializer for the Differential Evolution training model
        
        :param X_train: the training data to initialize the model with
        :param y_train: the target values for the training set
        :param n_nodes_per_layer: the number of nodes in each hidden layer
        :param n_hidden_layers: the number of hidden layers
        :param population: the population size for the training algorithm
        :param scaling: the scaling factor applied to the difference of vectors
        :param binomial_crossover: the probability that a gene will crossover from the donor
        :param is_classifier: whether the model is for a classifier
        """
        self.nets: list[ffNN.ffNN] = [None] * population
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.scaling = scaling
        self.binomial_crossover = binomial_crossover
        self.is_classifier = is_classifier
        self.initialize_model(X_train, y_train)

    def initialize_model(self, X_train, y_train):
        """initializer for the model, primarily for use in the tenfold crossvalidation
        
        :param X_train: the training data
        :param Y_train: the target values
        """
        # reset the lists
        self.nets: list[ffNN.ffNN] = [None] * len(self.nets)
        self.perf = None
        self.X_train = X_train
        self.y_train = y_train
        # initialize the neural nets
        for i in range(len(self.nets)):
            self.nets[i] = ffNN.ffNN(X_train, y_train, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
            self.nets[i].weights = [np.random.uniform(low=-1.0, high=1.0, size=self.nets[i].weights[n].shape) for n in
                                    range(self.nets[i].layers - 1)]
            self.nets[i].biases = [np.random.uniform(low=-1.0, high=1.0, size=self.nets[i].biases[n].shape) for n in
                                   range(self.nets[i].layers - 1)]

    def _generate_donor(self, x1, x2, x3):
        """generates a donor vector using x1 + scaling * (x2 - x3)
        
        :param x1: the x1 vector in the equation
        :param x2: the x2 vector in the equation
        :param x3: the x3 vector in the equation
        :return:
        """
        return [a + b for a, b in zip(x1, [self.scaling * (c - d) for c, d in zip(x2, x3)])]

    def _generate_offspring(self, x, donor):
        """Uses the binomial crossover probability to create an offspring vector with x and the donor
        
        :param x: the parent (?) vector
        :param donor: the donor vector
        :return:
        """
        # using the probability defined by the binomail crossover value, generate a selector and a complement set
        selector = [np.random.uniform(size=x[i].shape) > self.binomial_crossover for i in range(len(x))]
        antiselector = [selector[i] == False for i in range(len(x))]
        # we can use math to generate the offspring vector since True is treated as 1 and False is treated as 0
        offspring = [((x[i] * selector[i]) + (donor[i] * antiselector[i])) for i in range(len(x))]
        return offspring

    def train(self, X_test, Y_test):
        """the training method for the algorithm
        
        :param X_test: the vector set used to test for performance
        :param Y_test: the target set
        """
        bestperf = 0
        epochs = 0
        while epochs < 10:
            epochs += 1
            self._train()
            pred = self.predict(X_test)
            if self.is_classifier:
                perf = 1 - self._performance(pred, Y_test)
            else:
                perf = 1 / self._performance(pred, Y_test)
            if perf > bestperf:
                bestperf = perf
                epochs = 0

    def _train(self):
        """the real training algorithm
        """
        n = len(self.nets)
        next_perf = np.zeros(n)
        next_nets = [None] * n
        # if we don't have a performance array then we need to generate one
        if self.perf is None:
            self.perf = np.zeros(n)
            # generate the performance values for the neural nets in our population
            for i in range(n):
                Y_pred = self.nets[i].predict(self.X_train)
                self.perf[i] = self._performance(Y_pred, self.y_train)
        # get the indices of the sorted performance values
        sorted_indices = np.argsort(self.perf)
        for i in range(n):
            # find the best performance for the target, or else the second best if the current index is the best
            if sorted_indices[0] == i:
                target_index = sorted_indices[1]
            else:
                target_index = sorted_indices[0]
            # generate a list of valid indices for our difference vector candidates
            if target_index > i:
                valid_indices = list(range(i)) + list(range(i + 1, target_index)) + list(range(target_index + 1, n))
            else:
                valid_indices = list(range(target_index)) + list(range(target_index + 1, i)) + list(range(i + 1, n))
            # choose the difference vector indices from our list
            indices = np.random.choice(valid_indices, 2, replace=False)
            # generate our donor vectors
            donor_weights = self._generate_donor(self.nets[target_index].weights, self.nets[indices[0]].weights,
                                                 self.nets[indices[1]].weights)
            donor_biases = self._generate_donor(self.nets[target_index].biases, self.nets[indices[0]].biases,
                                                self.nets[indices[1]].biases)
            # generate our offspring vectors
            offspring_weights = self._generate_offspring(self.nets[i].weights, donor_weights)
            offspring_biases = self._generate_offspring(self.nets[i].biases, donor_biases)
            # initialize a ffNN and set its weights to the offspring values
            offspring_net = ffNN.ffNN(self.X_train, self.y_train, self.nets[0].weights[0].shape[0],
                                      self.nets[0].layers - 2, self.is_classifier)
            offspring_net.weights = offspring_weights
            offspring_net.biases = offspring_biases
            # get the performance of the offspring net
            pred = offspring_net.predict(self.X_train)
            offspring_perf = self._performance(pred, self.y_train)
            # if it is an improvement, add it. Otherwise, throw it away
            if offspring_perf < self.perf[i]:
                next_nets[i] = offspring_net
                next_perf[i] = offspring_perf
            else:
                next_nets[i] = self.nets[i]
                next_perf[i] = self.perf[i]
        # set the current batch of neural nets and performances to what we just worked on
        self.nets = next_nets
        self.perf = next_perf

    def _performance(self, Y_pred, Y_true):
        """Helper function for getting performance
        
        :param Y_pred: the target values that were predicted by the model
        :param Y_true: the real target values
        """
        # if it is a classifier, then it uses zero one loss, otherwise it does mean squared error
        if self.is_classifier:
            results = np.mean(Y_pred != Y_true)
        else:
            results = np.mean(np.square(Y_pred - Y_true))
        return results

    def predict(self, X):
        # if we don't have a performance array then we need to generate one
        if self.perf is None:
            print("Differential Evolution population requires training.")
            return None
        # uses the best performing net to predict the given feature vector set
        return self.nets[np.argmin(self.perf)].predict(X)
