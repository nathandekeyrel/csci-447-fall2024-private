import numpy as np


def ffNNClassification():
    pass

class ffNNRegression:
    def __init__(self, n_input, n_hidden, n_hidden_layers, n_output=1):
        #initialize the neural network nodes here
        self.X = None
        self.Y = None
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

    def train(self, X, Y, epochs, batchsize, l):
        for _ in range(epochs):
            #find some way to select the X and Y values for the batches
            indices = np.random.choice(X.shape[0], batchsize, replace=False)
            X_t = np.array(X[indices])
            Y_t = np.array(Y[indices])
            self._train(X_t, Y_t, l)
        pass

    def _train(self, X, Y, l):
        dt = []
        n = len(X)
        for mat in self.weights:
            dt.append(np.zeros((len(mat), 1)))
        for i in range(n):
            outs = self.feedforward(X[i])
            ds = self.backprop(Y[i], outs[-1][0][0], l)
            for i in range(len(dt)):
                dt[i] += ds[i]
        
        for i in range(len(dt)):
            self.weights[i] += np.dot(dt[i], outs[i].T) / n
            self.biases[i] += dt[i] / n

    def predict(self, X):
        return [self.feedforward(x)[-1][0][0] for x in X]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    try:
        r = x * (1 - x)
    except:
        r = 0
    return r