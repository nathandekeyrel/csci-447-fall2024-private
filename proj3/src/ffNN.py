import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ffNNClassification:
    def __init__(self, n_input, n_hidden, n_hidden_layers, n_output):
        """Initialize the neural network

        :param n_input: number of input nodes
        :param n_hidden: number of nodes in hidden layer
        :param n_hidden_layers: number of hidden layers
        :param n_output: number of output nodes
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.n_output = n_output
        self.layers = n_hidden_layers + 2
        self.outputs = None
        self.biases = []
        self.weights = []
        self.weight_velocities = []
        self.bias_velocities = []
        layer_sizes = []
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.arange(self.n_output).reshape(-1, 1))

        if n_hidden_layers == 0:
            layer_sizes.append([n_output, n_input])  # direct input -> output
        else:
            layer_sizes.append([n_hidden, n_input])  # input -> first hidden
            for _ in range(n_hidden_layers - 1):
                layer_sizes.append([n_hidden, n_hidden])  # hidden -> hidden
            layer_sizes.append([n_output, n_hidden])  # last hidden -> output

        # initialize small weights and biases to help with vanishing gradient problem
        self.weights = [(np.random.rand(x, y) + -0.5) * 0.002 for x, y in layer_sizes]
        self.biases = [(np.random.rand(x, 1) + -0.5) * 0.002 for x, _ in layer_sizes]

        # initialize zero arrays in similar type and shape to weight and bias vectors
        self.weight_velocities = [np.zeros_like(w) for w in self.weights]
        self.bias_velocities = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def softmax(x):
        """Converts a vector of real numbers into a probability distribution over multiple classes.
        The probabilities of each value are proportional to the relative scale of each value in the vector.

        :param x: input array of real values
        :return: vector of probability scores that sum to 1
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def feedforward(self, x: np.ndarray):
        """Performs forward propagation through the neural network, computing activations
        at each layer using the current weights and biases. The hidden layers use sigmoid,
        the output layer uses softmax

        :param x: input features from dataset
        :return: probability distribution over classes
        """
        self.outputs = []
        self.outputs.append(x.reshape(-1, 1))

        if self.n_hidden_layers == 0:  # if no hidden layers
            activation = np.dot(self.weights[0], self.outputs[0]) + self.biases[0]
            self.outputs.append(self.softmax(activation))
        else:  # if > 0 hidden layers
            for i in range(self.layers - 2):
                activation = np.dot(self.weights[i], self.outputs[i]) + self.biases[i]
                self.outputs.append(sigmoid(activation))

            activation = np.dot(self.weights[-1], self.outputs[-1]) + self.biases[-1]
            self.outputs.append(self.softmax(activation))

        return self.outputs

    def backprop(self, y_onehot, learning_rate):
        """Performs backpropagation through the neural network to compute weight updates
        based on prediction error. For networks with hidden layers, uses the chain rule to
        calculate gradients at each layer, working backwards from the output.

        :param y_onehot: one-hot encoded target values from dataset
        :param learning_rate: step size for gradient update
        :return: tuple of weight updates and deltas for each layer
        """

        if self.n_hidden_layers == 0:  # if no hidden layers
            output_error = self.outputs[-1] - y_onehot.reshape(-1, 1)
            weight_update = learning_rate * np.dot(output_error, self.outputs[0].T)
            return [weight_update], [output_error]
        else:  # if > 0 hidden layers
            deltas = []
            weight_updates = []

            # calculate initial output layer error
            output_error = self.outputs[-1] - y_onehot.reshape(-1, 1)
            delta = output_error  # delta is error at output layer
            deltas.insert(0, delta)

            for i in range(len(self.weights) - 1, 0, -1):  # backprop through layers
                delta = np.dot(self.weights[i].T, delta) * d_sigmoid(self.outputs[i])
                deltas.insert(0, delta)

            # compute weight updates for all layers using deltas
            for i in range(len(self.weights)):
                weight_update = learning_rate * np.dot(deltas[i], self.outputs[i].T)
                weight_updates.append(weight_update)

            return weight_updates, deltas

    def train(self, X, y, epochs, batchsize, learning_rate, momentum):
        """Trains the neural network using mini-batch gradient descent.
        For each epoch, shuffles data randomly and processes it into
        batches to update weights and biases using backpropagation.

        :param X: input features from dataset
        :param y: target values from dataset
        :param epochs: number of complete passes through training dataset
        :param batchsize: number of samples processed before updating weights
        :param learning_rate: step size for gradient descent
        :param momentum: coefficient for momentum to escape local minima
        """
        y_onehot = self.encoder.transform(y.reshape(-1, 1))

        for _ in range(epochs):
            # prevent learning order bias by shuffling data
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_onehot_shuffled = y_onehot[permutation]

            # process data in mini-batches to update network weights
            for i in range(0, X.shape[0], batchsize):
                # extract current batch using slice
                X_batch = X_shuffled[i:i + batchsize]
                y_batch = y_onehot_shuffled[i:i + batchsize]
                # train on current batch using helper method
                self._train(X_batch, y_batch, learning_rate, momentum)

    def _train(self, X, y_onehot, learning_rate, momentum):
        """Helper method that performs training on a batch of samples. Updates network
        weights and biases using accumulated gradients over the batch with momentum.

        :param X: input features from dataset
        :param y_onehot: target values from dataset
        :param learning_rate: step size for gradient descent
        :param momentum: coefficient for momentum to escape local minima
        """
        n = len(X)  # batch size
        # initialize arrays to accumulate gradients over batch
        weight_updates_sum = [np.zeros_like(w) for w in self.weights]
        bias_updates_sum = [np.zeros_like(b) for b in self.biases]

        for i in range(n):
            self.feedforward(X[i])  # forward pass to compute activations
            weight_updates, deltas = self.backprop(y_onehot[i], learning_rate)  # backwards pass to compute gradients

            # accumulate updates over batch
            for ii in range(len(self.weights)):
                weight_updates_sum[ii] += weight_updates[ii]
                bias_updates_sum[ii] += deltas[ii]

        # update weights and biases using averaged gradients and momentum
        for i in range(len(self.weights)):
            weight_gradient = weight_updates_sum[i] / n
            bias_gradient = bias_updates_sum[i] / n

            # update velocities using momentum term and current gradients
            self.weight_velocities[i] = momentum * self.weight_velocities[i] - weight_gradient
            self.bias_velocities[i] = momentum * self.bias_velocities[i] - bias_gradient

            # apply updates using computed velocities
            self.weights[i] += self.weight_velocities[i]
            self.biases[i] += self.bias_velocities[i]

    def predict(self, X):
        """Makes class predictions for multiple input samples using the trained network.
        For each sample, performs forward propagation and selects the class with the highest
        probability from the softmax output.

        :param X: input features from dataset
        :return: class with the highest probability
        """
        predictions = []
        for x in X:
            outputs = self.feedforward(x)  # get output probs for current sample
            output = outputs[-1]  # get final layer outputs (the prob dist from softmax)
            predicted_class = np.argmax(output)  # select class with highest prob
            predictions.append(predicted_class)
        return predictions


class ffNNRegression:
    def __init__(self, X, Y, n_input, n_hidden, n_hidden_layers, n_output=1):
        #initialize the neural network nodes here
        self.X = X
        self.Y = Y
        self.layers = n_hidden_layers + 2
        self.weights = []
        self.biases = []
        self.outputs = None
        dims = [[0, n_input]]
        i = -1
        for i in range(self.layers - 2):
            dims[i][0] = n_hidden
            dims.append([0, n_hidden])
        dims[i + 1][0] = n_output
        self.weights = [(np.random.rand(x, y) + -0.5) * 0.002 for x, y in dims]
        self.biases = [(np.random.rand(x, 1) + -0.5) * 0.002 for x, _ in dims]
        self.dw_prev = [np.zeros((x, y)) for x, y in dims]
        self.db_prev = [np.zeros((x, 1)) for x, _ in dims]
        pass

    def feedforward(self, x : np.ndarray):
        """ feed the feature vector through the network and receive the output
        
        :param x: the feature vector
        :return: the output from the feature vector
        """
        # get the dot product of the input and the input to hidden layer weight matrix
        self.outputs = []
        self.outputs.append(x.reshape(len(x), 1))
        i = -1 #initialize to -1 in case the number of layers is less than 3
        for i in range(self.layers - 2):
            activation = np.dot(self.weights[i], self.outputs[i]) + self.biases[i]
            self.outputs.append(sigmoid(activation))
        #special case for the last node. Increment i by one because python doesn't have normal-ass for loops
        i += 1
        activation = np.dot(self.weights[i], self.outputs[i]) + self.biases[i]
        self.outputs.append(activation)
        return self.outputs

    def backprop(self, y, p, l):
        errors = [None for _ in range(self.layers - 1)]
        delta = [None for _ in range(self.layers - 1)]
        #since the last node is a special case, I'm just going to do this instead writing a bunch of fucking branching instructions
        errors[-1] = np.array([[y - p]])
        delta[-1] = errors[-1] * l
        for i in range(self.layers - 3, -1, -1):
            #get the e value for the current layer via e_i = e_i+1 dot W_i+1
            errors[i] = np.dot(errors[i+1], self.weights[i+1])
            #get the delta values per layer
            delta[i] = d_sigmoid(self.outputs[i+1]) * errors[i].T * l
        return delta

    def train(self, epochs, batchsize, l, a):
        for _ in range(epochs):
            #find some way to select the X and Y values for the batches
            indices = np.random.choice(self.X.shape[0], batchsize, replace=False)
            X_t = np.array(self.X[indices])
            Y_t = np.array(self.Y[indices])
            self._train(X_t, Y_t, l, a)
        pass

    def _train(self, X_t, Y_t, l, a):
        dw = [] # list that holds our changes for our weights
        db = [] # list that holds our changes for our biases
        n = len(X_t) # number of training examples
        # initialize dt to zero with the dimensions we are expecting from the deltas reported by backprop
        for mat in self.weights:
            dw.append(np.zeros(mat.shape))
            db.append(np.zeros((len(mat), 1)))
        # iterate through each training example and add the results from backprop to dt
        for i in range(n):
            outs = self.feedforward(X_t[i])
            ds = self.backprop(Y_t[i], outs[-1][0][0], l)
            for i in range(self.layers - 1):
                dw[i] += np.dot(ds[i], outs[i].T)
                db[i] += ds[i]
        for i in range(self.layers - 1):
            dw[i] /= n
            db[i] /= n
            self.weights[i] += (dw[i] + (a * self.dw_prev[i]))
            self.biases[i] += (db[i] + (a * self.db_prev[i]))
        self.dw_prev = dw
        self.db_prev = db

    def predict(self, X):
        return [self.feedforward(x)[-1][0][0] for x in X]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)