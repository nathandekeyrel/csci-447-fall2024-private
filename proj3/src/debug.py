import preprocess as pr
import ffNN as fn
import evaluating as ev
import tenfoldcv as xv
import utils as ut
import numpy as np
from numpy import mean, square

np.seterr(all='raise')

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

X_train, Y_train, X_test, Y_test = ut.generateTestData(X, Y)

re = fn.ffNNRegression(X_train, Y_train, len(X[0]), len(X[0]) * 2, 3)
# cl = fn.ffNNClassification(len(X[0]), len(X[0]) * 2, 1, len(Y[0]))

# f = open("output/soybean.csv", 'w')
ept = 100
epochs = 0
while True:
# for i in range(1000):
  epochs += ept
  re.train(ept, 100, .0001, 1)
  p = re.predict(X_test)
  # cl.train(X_train, Y_train, ept, 10, 0.01, 0.0)
  # p = cl.predict(X_test)
  print("epoch: " + str(epochs))
  print(ev.mse(Y_test, p))
  print(r2(Y_test, p))
  # Y_true = np.array([np.argmax(y) for y in Y_test])
  # f.write(str(epochs) + "," + str(ev.zero_one_loss(Y_true, p)) + "\n")