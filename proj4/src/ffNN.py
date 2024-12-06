import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ffNN:
    def __init__(self, X, y, n_hidden, n_hidden_layers, is_classifier):
        self.X = X
        self.layers = n_hidden_layers + 2
        self.is_classifier = is_classifier
        self.weights = []
        self.biases = []
        self.outputs = None
        n_input = X.shape[1]
        self.y = y.reshape(-1, 1)
        if is_classifier:
            encoder = OneHotEncoder(sparse_output=False)
            encoder.fit(self.y)
            self.y = encoder.transform(self.y)
        n_output = self.y.shape[1]
        dims = [[0, n_input]]
        i = -1
        for i in range(self.layers - 2):
            dims[i][0] = n_hidden
            dims.append([0, n_hidden])
        dims[i + 1][0] = n_output
        self.weights = [(np.random.rand(x, y) + -0.5) * 0.002 for x, y in dims]
        self.biases = [(np.random.rand(x, 1) + -0.5) * 0.002 for x, _ in dims]

    def feedforward(self, x):
        self.outputs = [None for _ in range(self.layers)]
        self.outputs[0] = x.reshape(-1, 1)
        activation = self.weights[0].dot(self.outputs[0]) + self.biases[0]
        for i in range(1, self.layers - 1):
            self.outputs[i] = sigmoid(activation)
            activation = np.dot(self.weights[i], self.outputs[i]) + self.biases[i]
        self.outputs[-1] = self.softmax(activation) if self.is_classifier else activation
        return self.outputs[-1]

    def _predict(self, X):
        return np.array([self.feedforward(x) for x in X])

    def predict(self, X):
        r = self._predict(X)
        if self.is_classifier:
            r = r.argmax(axis=1)
        return r.flatten()

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


def sigmoid(x):
    """Sigmoid activation function that maps any real number to [0,1] range.

    :param x: input value or numpy array to transform
    :return: value between 0 and 1
    """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """Derivative of sigmoid function. Used in backpropagation to
    compute gradients.Its derivative is f'(x) = f(x) * (1 - f(x)).

    :param x: input value already passed through the sigmoid function
    :return: the derivative value at that point
    """
    return x * (1 - x)
