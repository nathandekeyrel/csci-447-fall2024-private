import ffNN as fn
import numpy as np
from copy import deepcopy as cp

class ffNNbackprop:
  def __init__(self, X, Y, nhl, npl, bs, lr, m, is_cl):
    self.net = fn.ffNN(X, Y, npl, nhl, is_cl)
    self.dw_vel = [np.zeros(mat.shape) for mat in self.net.weights]
    self.db_vel = [np.zeros(mat.shape) for mat in self.net.biases]
    self.bs = bs
    self.lr = lr
    self.m = m
  
  def train(self, X_test, Y_test):
    bestperf = 0
    bestweights = cp(self.net.weights), cp(self.net.biases)
    epochs = 0
    while epochs < 25:
      epochs += 1
      indices = np.random.choice(self.net.X.shape[0], self.bs, replace=False)
      X_t = self.net.X[indices]
      Y_t = self.net.Y[indices]
      self._train(X_t, Y_t)
      perf = self.performance(X_test, Y_test)
      if perf > bestperf:
        bestperf = perf
        bestweights = cp(self.net.weights), cp(self.net.biases)
        epochs = 0
    self.net.weights, self.net.biases = cp(bestweights)
  
  def _train(self, X_t, Y_t):
    dw = [np.zeros(mat.shape) for mat in self.net.weights]
    db = [np.zeros(mat.shape) for mat in self.net.biases]
    for i in range(X_t.shape[0]):
      self.net.feedforward(X_t[i])
      deltas = self.backprop(Y_t[i])
      for i in range(self.net.layers - 1):
        dw[i] += np.dot(deltas[i], self.net.outputs[i].T)
        db[i] += deltas[i]
    for i in range(self.net.layers - 1):
      dw[i] *= (self.lr / X_t.shape[0])
      db[i] *= (self.lr / X_t.shape[0])
      dw[i] += (self.m * self.dw_vel[i])
      db[i] += (self.m * self.db_vel[i])
      self.net.weights[i] += dw[i]
      self.net.biases[i] += db[i]
    self.dw_vel = dw
    self.db_vel = db
  
  def backprop(self, y_true):
    deltas = [None] * (self.net.layers - 1)
    deltas[-1] = y_true.reshape(-1, 1) - self.net.outputs[-1]
    for i in range(2, self.net.layers):
      deltas[-i] = np.dot(self.net.weights[-i + 1].T, deltas[-i + 1]) * fn.d_sigmoid(self.net.outputs[-i])
    return deltas
  
  def performance(self, X_test, Y_true):
    Y_pred = self.net.predict(X_test)
    if self.net.is_classifier:
      perf = np.mean(Y_true == Y_pred)
    else:
      perf = 1 / np.mean(np.square(Y_true - Y_pred))
    return perf