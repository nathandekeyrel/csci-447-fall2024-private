import preprocess as pr
import ffNN as fn
import evaluating as ev
import tenfoldcv as xv
import utils as ut
import tuning as tu
import nutuning as ntu
import numpy as np
from numpy import mean, square

# np.seterr(all='raise')
test = "tuning"

def r2(y_true, y_pred):
  m = mean(y_true)
  e2_s = 0
  v2_s = 0
  for i in range(len(y_true)):
    e2_s += square(y_true[i] - y_pred[i])
    v2_s += square(y_true[i] - m)
  r = 1 - (e2_s / v2_s)
  return r

X, Y = pr.preprocess_data("data/machine.data")

if test == "tuning":
  learning_rate, batch_size, n_hidden, momentum = ntu.tuneFFNNRegression(X, Y, 3)
  print(learning_rate, batch_size, n_hidden, momentum)
  pass

elif test == "reg":
  X_train, Y_train, X_test, Y_test = ut.generateTestData(X, Y)

  re = fn.ffNNRegression(X_train, Y_train, len(X[0]), len(X[0]) * 2, 5)
  # cl = fn.ffNNClassification(len(X[0]), len(X[0]) * 2, 1, len(Y[0]))



  steps = 0
  bestresults = np.square(np.max(Y_test) - np.min(Y_test))
  prevresults = bestresults
  eps = 1
  epochs = 0
  while steps < 20:
    epochs += eps
    steps += 1
    re.train(eps, len(X_train), .2, .5)
    p = re.predict(X_test)
    results = ev.mse(Y_test, p)
    if results < bestresults:
      bestresults = results
      steps = 0
    print("epochs: " + str(epochs))
    print(results)
  print(epochs)
  print(bestresults)

