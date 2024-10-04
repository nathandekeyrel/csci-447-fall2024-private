import knn as knn
import editedKNN as eknn
import random as r
import time as t
import tenfoldcv as kfxv
import numpy as np
import tuning as tu

r.seed(t.time())
lc: list[list] = []
lr: list[list] = []

# generate list of n vectors with random x and y values, and the associated class
n = 100
for i in range(n):
    x = r.random()
    y = r.random()
    # classify those items as group 0 if x > y, and group 1 if x <= y
    c = 0 if x > y else 1
    lc.append([x, y, c])

print(tu.tuneKNN(lc, [0, 1]))

""" n = 100
for i in range(n):
  x = i
  y = (r.random() * np.sqrt(n)) + (i - np.sqrt(n) / 2)
  lr.append([x, y])

lr = kfxv.kfold(lr, 10)
trr = lr.pop(0)
lr = kfxv.mergedata(lr)

regression = eknn.EKNNTrueRegression(lr, trr, 5, 5)

for x in trr:
  print(str(x[0]), " ", str(regression.predict(x, 5, ))) """

""" 
print(np.array([[np.sqrt(np.sum(x - y) ** 2) for y in ys] for x in xs]))
print(np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(xs, ys)]))) """
