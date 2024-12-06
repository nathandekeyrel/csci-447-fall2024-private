import preprocess as pr
import ffNN as fn
import evaluating as ev
import tenfoldcv as xv
import utils as ut
import tuning as tu
import nutuning as ntu
import numpy as np
import backprop as bp
from DifferentialEvolution import DifferentialEvolution as diff
from ParticleSwarmOptimization import PSO
from copy import deepcopy as cp
from numpy import mean, square
import time
import multiprocessing as mp

# np.seterr(all='raise')
test = "fulltuningmp"

Xr, Yr = pr.preprocess_data("data/machine.data")

Xc, Yc = pr.preprocess_data("data/breast-cancer-wisconsin.data")

def mptune(path, nodes, is_classifier, isPSO):
  if path == "forestfires":
    data = "data/" + path + ".csv"
  else:
    data = "data/" + path + ".data"
  if isPSO:
    output = "outputs/pso" + path + ".csv"
  else:
    output = "outputs/de" + path + ".csv"
  X, Y = pr.preprocess_data(data)
  file = open(output, "w")
  file.write(path)
  file.flush()
  if isPSO:
    file.write(str(ntu.tunePSO(X, Y, nodes, 1, is_classifier)))
  else:
    file.write(str(ntu.tuneDE(X, Y, nodes, 1, is_classifier)))

if test == "fulltuningmp":
  paths = ["abalone", "forestfires", "machine", "breast-cancer-wisconsin", "glass", "soybean-small"]
  nodes = [13, 25, 30, 5, 7, 38]
  is_classifier = [False, False, False, True, True, True]
  demps = [mp.Process(target=mptune, args=(paths[0], nodes[0], is_classifier[0], False))]
  psomps = [mp.Process(target=mptune, args=(paths[0], nodes[0], is_classifier[0], True)),
            mp.Process(target=mptune, args=(paths[2], nodes[2], is_classifier[2], True)),
            mp.Process(target=mptune, args=(paths[5], nodes[5], is_classifier[5], True))]
  for demp in demps:
    demp.start()
  for psomp in psomps:
    psomp.start()
  
  for psomp in psomps:
    psomp.join()
  for demp in demps:
    demp.join()

if test == "fulltuning2":
  psofile = open("outputs/pso.csv", "w")
  defile = open("outputs/de.csv", "w")
  paths = ["data/abalone.data", "data/forestfires.csv", "data/machine.data", "data/breast-cancer-wisconsin.data", "data/glass.data", "data/soybean-small.data"]
  nodes = [13, 25, 30, 5, 7, 38]
  for i in range(3):
    X, Y = pr.preprocess_data(paths[i])
    psofile.write(paths[i])
    psofile.write("\n")
    psofile.flush()
    psofile.write(str(ntu.tunePSO(X, Y, nodes[i], 1, False)))
    psofile.write("\n")
    psofile.flush()
    defile.write(paths[i])
    defile.write("\n")
    psofile.flush()
    defile.write(str(ntu.tuneDE(X, Y, nodes[i], 1, False)))
    defile.write("\n")
    psofile.flush()
  for i in range(3, 6):
    X, Y = pr.preprocess_data(paths[i])
    psofile.write(paths[i])
    psofile.write("\n")
    psofile.flush()
    psofile.write(str(ntu.tunePSO(X, Y, nodes[i], 1, True)))
    psofile.write("\n")
    psofile.flush()
    defile.write(paths[i])
    defile.write("\n")
    psofile.flush()
    defile.write(str(ntu.tuneDE(X, Y, nodes[i], 1, True)))
    defile.write("\n")
    psofile.flush()

if test == "detrain":
  X, Y, X_test, Y_test = ut.generateTestData(Xr, Yr)
  de = diff(X, Y, 30, 1, 50, 0.5, 0.5, False)
  start = time.time()
  de.train(X_test, Y_test)
  end = time.time()
  print(end - start)

if test =="psotrain":
  X, Y, X_test, Y_test = ut.generateTestData(Xr, Yr)
  pso = PSO(X, Y, 30, 1, 50, 0.5, 0.5, 0.5, False)
  start_time = time.time()
  pso.train(X_test, Y_test)
  end_time = time.time()
  print(end_time - start_time)

if test == "psotuning":
  is_classifier = False
  if is_classifier:
    X, Y = Xc, Yc
  else:
    X, Y = Xr, Yr
  print(ntu.tuneDE(X, Y, 30, 1, is_classifier))

if test == "Diff":
  is_classifier = True
  if is_classifier:
    X, Y, X_test, Y_test = ut.generateTestData(Xc, Yc)
  else:
    X, Y, X_test, Y_test = ut.generateTestData(Xr, Yr)
  model = PSO(X, Y, 5, 1, 20, 0.8, 0.7, 0.3, is_classifier)
  model.train(X_test, Y_test)
  pred = model.predict(X_test)
  if is_classifier:
    print(ev.zero_one_loss(Y_test, pred))
  else:
    print(ev.mse(Y_test, pred))
  pass

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