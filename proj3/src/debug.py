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
test = "printtuning"

X, Y = pr.preprocess_data("data/machine.data")

Xc, Yc = pr.preprocess_data("data/breast-cancer-wisconsin.data")

if test == "printtuning":
  file = open("output/tunings.csv", "w")
  file.write("name, hidden layers, learning rate, batch size, nodes in each hidden layer, momentum\n")
  file.flush()
  paths = ["machine.data", "forestfires.csv", "abalone.data"]
  for path in paths:
    X, Y = pr.preprocess_data("data/" + path)
    learning_rate, batch_size, n_hidden, momentum = ntu.tuneFFNNRegression(X, Y, 0)
    file.write('%s, 0, %f, %d, %d, %f\n'%(path, learning_rate, batch_size, n_hidden, momentum))
    file.flush()
    learning_rate, batch_size, n_hidden, momentum = ntu.tuneFFNNRegression(X, Y, 1)
    file.write('%s, 1, %f, %d, %d, %f\n'%(path, learning_rate, batch_size, n_hidden, momentum))
    file.flush()
    learning_rate, batch_size, n_hidden, momentum = ntu.tuneFFNNRegression(X, Y, 2)
    file.write('%s, 2, %f, %d, %d, %f\n'%(path, learning_rate, batch_size, n_hidden, momentum))
    file.flush()
  file.close()

if test == "classcv":
  out = xv.tenfoldcrossvalidationC(Xc, Yc, 3, len(Xc[0]), 20, 1.0, 0.0)
  print(np.mean([ev.zero_one_loss(yt, yp) for yt, yp in out]))

if test == "fullreg":
  learning_rate, batch_size, n_hidden, momentum = ntu.tuneFFNNRegression(X, Y, 3)
  out = xv.tenfoldcrossvalidationR(X, Y, 3, n_hidden, batch_size, learning_rate, momentum)
  print(np.mean([ev.mse(yt, yp) for yt, yp in out]))

if test == "regcv":
  out = xv.tenfoldcrossvalidationR(X, Y, 3, 6, 2718, 0.265686, 0.42882)
  print(np.mean([ev.mse(yt, yp) for yt, yp in out]))
  pass

if test == "regtuning":
  learning_rate, batch_size, n_hidden, momentum = ntu.tuneFFNNRegression(X, Y, 3)
  print(learning_rate, batch_size, n_hidden, momentum)
  pass

elif test == "reg":
  X_train, Y_train, X_test, Y_test = ut.generateTestData(X, Y)

  re = fn.ffNNRegression(X_train, Y_train, len(X[0]), len(X[0]) * 2, 5)

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