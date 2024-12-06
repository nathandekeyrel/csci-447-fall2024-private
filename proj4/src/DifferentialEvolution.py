import ffNN
import numpy as np
from copy import deepcopy as cp
import threading as th
import multiprocessing as mp

class DifferentialEvolution:
    def __init__(self, X_train, Y_train, n_nodes_per_layer, n_hidden_layers, population, scaling, binomial_crossover, is_classifier):
        self.nets : list[ffNN.ffNN] = [None] * population
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.scaling = scaling
        self.binomial_crossover = binomial_crossover
        self.is_classifier = is_classifier
        self.perf = None
        self.initialize_model(X_train, Y_train)
    
    def initialize_model(self, X_train, Y_train):
        self.nets : list[ffNN.ffNN] = [None] * len(self.nets)
        self.perf = None
        self.X_train = X_train
        self.Y_train = Y_train
        for i in range(len(self.nets)):
            self.nets[i] = ffNN.ffNN(X_train, Y_train, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
            self.nets[i].weights = [np.random.uniform(low=-1.0, high=1.0, size=self.nets[i].weights[n].shape) for n in range(self.nets[i].layers - 1)]
            self.nets[i].biases = [np.random.uniform(low=-1.0, high=1.0, size=self.nets[i].biases[n].shape) for n in range(self.nets[i].layers - 1)]
    
    def _generate_donor(self, x1, x2, x3):
        return [a + b for a, b in zip(x1, [self.scaling * (c - d) for c, d in zip(x2, x3)])]
    
    def _generate_offspring(self, x, donor):
        selector = [np.random.uniform(size=x[i].shape) > self.binomial_crossover for i in range(len(x))]
        antiselector = [selector[i] == False for i in range(len(x))]
        offspring = [((x[i] * selector[i]) + (donor[i] * antiselector[i])) for i in range(len(x))]
        return offspring
    
    def train(self, X_test, Y_test):
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
        n = len(self.nets)
        next_perf = np.zeros(n)
        next_nets = [None] * n
        # if we don't have a performance array then we need to generate one
        if self.perf is None:
            self.perf = np.zeros(n)
            # generate the performance values for the neural nets in our population
            for i in range(n):
                Y_pred = self.nets[i].predict(self.X_train)
                self.perf[i] = self._performance(Y_pred, self.Y_train)
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
            donor_weights = self._generate_donor(self.nets[target_index].weights, self.nets[indices[0]].weights, self.nets[indices[1]].weights)
            donor_biases = self._generate_donor(self.nets[target_index].biases, self.nets[indices[0]].biases, self.nets[indices[1]].biases)
            # generate our offspring vectors
            offspring_weights = self._generate_offspring(self.nets[i].weights, donor_weights)
            offspring_biases = self._generate_offspring(self.nets[i].biases, donor_biases)
            # initialize a ffNN and set its weights to the offspring values
            offspring_net = ffNN.ffNN(self.X_train, self.Y_train, self.nets[0].weights[0].shape[0], self.nets[0].layers - 2, self.is_classifier)
            offspring_net.weights = offspring_weights
            offspring_net.biases = offspring_biases
            # get the performance of the offspring net
            pred = offspring_net.predict(self.X_train)
            offspring_perf = self._performance(pred, self.Y_train)
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