import knn as knn
#import editedKNN as eknn
import random as r
import time as t
import tenfoldcv as kfxv
import numpy as np
#import tuning as tu

r.seed(t.time())
lc : list[list] = []
lr : list[list] = []

""" #generate list of n vectors with random x and y values, and the associated class
n = 100
for i in range(n):
  x = r.random()
  y = r.random()
  #classify those items as group 0 if x > y, and group 1 if x <= y
  c = 0 if x > y else 1
  lc.append([x, y, c]) """

""" x1 = np.array([0, 1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1, 0])

print() """

n = 100
X = np.array([i for i in range(n)])
Y = np.array([(r.random() * np.sqrt(n)) + (i - np.sqrt(n) / 2) for i in range(n)])

X, Y = kfxv.kfold(X, Y, 10)
Xt = X.pop(0)
Yt = Y.pop(0)
X = kfxv.mergedata(X)
Y = kfxv.mergedata(Y)

regression = knn.KNNRegression()
regression.fit(X, Y)
print(regression.predict(Xt, 5))
print(Yt)


""" 
print(np.array([[np.sqrt(np.sum(x - y) ** 2) for y in ys] for x in xs]))
print(np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(xs, ys)]))) """