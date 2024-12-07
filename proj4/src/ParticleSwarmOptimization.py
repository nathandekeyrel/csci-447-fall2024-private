import ffNN
import numpy as np
from copy import deepcopy as cp


class PSO:
    def __init__(self, X_train, y_train, n_nodes_per_layer, n_hidden_layers, population, inertia, cognitive_update_rate,
                 social_update_rate, is_classifier):
        """Initializer for the particle swarm optimization model
        
        :param X_train: the training data to initialize the model with
        :param y_train: the target values for the training set
        :param n_nodes_per_layer: the number of nodes in each hidden layer
        :param n_hidden_layers: the number of hidden layers
        :param population: the population size for the training algorithm
        :param inertia: how much of the velocity from the previous time slice is preserved
        :param cognitive_update_rate: the rate of impact that the personal best has on the agent
        :param social_update_rate: the rate of impact that the best of its neighbors has on the agent
        :param is_classifier: whether the model is for a classifier
        """
        self.nets: list[ffNN.ffNN] = [None] * population
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.inertia = inertia
        self.cognitive_update_rate = cognitive_update_rate
        self.social_update_rate = social_update_rate
        self.weight_velocities = None
        self.bias_velocities = None
        self.personal_bests = None
        self.personal_best_weights = None
        self.personal_best_biases = None
        self.is_classifier = is_classifier
        self.global_best_index: int = None
        self.initialize_model(X_train, y_train)

    def initialize_model(self, X, Y):
        self.X_train = X
        self.y_train = Y
        # initialize the weights for the neural nets
        for i in range(len(self.nets)):
            self.nets[i] = ffNN.ffNN(X, Y, self.n_nodes_per_layer, self.n_hidden_layers, self.is_classifier)
            self.nets[i].weights = [np.random.uniform(low=-1.0, high=1.0, size=self.nets[i].weights[n].shape) for n in
                                    range(self.nets[i].layers - 1)]
            self.nets[i].biases = [np.random.uniform(low=-1.0, high=1.0, size=self.nets[i].biases[n].shape) for n in
                                   range(self.nets[i].layers - 1)]
        # set the initial velocities to zero
        self.weight_velocities = [[np.zeros(mat.shape) for mat in self.nets[i].weights] for i in range(len(self.nets))]
        self.bias_velocities = [[np.zeros(mat.shape) for mat in self.nets[i].biases] for i in range(len(self.nets))]
        self.personal_best_weights = [cp(self.nets[i].weights) for i in range(len(self.nets))]
        self.personal_best_biases = [cp(self.nets[i].biases) for i in range(len(self.nets))]
        self.personal_bests = [self._performance(self.nets[i].predict(self.X_train), self.y_train) for i in
                               range(len(self.nets))]
        self.global_best_index = np.argmin(self.personal_bests)

    def _calculate_velocities(self):
        # i is the network index, j is the layer index
        # here, we set the velocities based on the update rule outlined in the paper (I'm not writing that here)
        weight_velocities = [
            [
                self.inertia * self.weight_velocities[i][j] +
                self.cognitive_update_rate * (self.personal_best_weights[i][j] - self.nets[i].weights[j]) +
                self.social_update_rate * (
                        self.personal_best_weights[self.global_best_index][j] - self.nets[i].weights[j])
                for j in range(self.nets[0].layers - 1)
            ]
            for i in range(len(self.nets))
        ]
        # this is for updating the bias
        bias_velocities = [
            [
                self.inertia * self.bias_velocities[i][j] +
                self.cognitive_update_rate * (self.personal_best_biases[i][j] - self.nets[i].biases[j]) +
                self.social_update_rate * (
                        self.personal_best_biases[self.global_best_index][j] - self.nets[i].biases[j])
                for j in range(self.nets[0].layers - 1)
            ]
            for i in range(len(self.nets))
        ]
        self.weight_velocities = weight_velocities
        self.bias_velocities = bias_velocities

    def _calculate_positions(self):
        # i is the network index, j is the layer index
        # update the position to x(t) <- x(t-1) + v(t)
        for i in range(len(self.nets)):
            self.nets[i].weights = [
                self.nets[i].weights[j] + self.weight_velocities[i][j]
                for j in range(self.nets[i].layers - 1)
            ]
        for i in range(len(self.nets)):
            self.nets[i].biases = [
                self.nets[i].biases[j] + self.bias_velocities[i][j]
                for j in range(self.nets[i].layers - 1)
            ]

    def _calculate_performances(self):
        # get the performance values for each unit and update them if they are improved for pbest
        for i in range(len(self.nets)):
            perf = self._performance(self.nets[i].predict(self.X_train), self.y_train)
            if perf <= self.personal_bests[i]:
                self.personal_bests[i] = perf
                self.personal_best_weights[i] = cp(self.nets[i].weights)
                self.personal_best_biases[i] = cp(self.nets[i].biases)
        # get the best performance in the list and update that
        self.global_best_index = np.argmin(self.personal_bests)

    def _train(self):
        # just iterate through all of the calculation functions
        self._calculate_velocities()
        self._calculate_positions()
        self._calculate_performances()

    def train(self, X_test, Y_test):
        bestperf = 0
        epochs = 0
        global_best_index = self.global_best_index
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
                global_best_index = self.global_best_index
        self.global_best_index = global_best_index

    def _performance(self, Y_pred, Y_true):
        # if it is a classifier, then it uses zero one loss, otherwise it does mean squared error
        if self.is_classifier:
            results = np.mean(Y_pred != Y_true)
        else:
            results = np.mean(np.square(Y_pred - Y_true))
        return results

    def predict(self, X):
        return self.nets[self.global_best_index].predict(X)
