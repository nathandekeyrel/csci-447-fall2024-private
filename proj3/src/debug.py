import preprocess as pr
import ffNN as fn
import evaluating as ev
import tenfoldcv as xv
from numpy import mean, square

def r2(y_true, y_pred):
  m = mean(y_true)
  e2_s = 0
  v2_s = 0
  for i in range(len(y_true)):
    e2_s += square(y_true[i] - y_pred[i])
    v2_s += square(y_true[i] - m)
  r = 1 - (e2_s / v2_s)
  return r

X, Y = pr.preprocess_data("data/forestfires.csv")

xv.

re = fn.ffNNRegression(len(X[0]), len(X[0]) * 2, 1)

while True:
  re.train(X, Y, 2000, 15, .05)
  p = re.predict(X)
  print(ev.mse(Y, p))
  print(r2(Y, p))