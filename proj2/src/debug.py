import knn as knn
import editedKNN as eknn
import random as r
import time as t
import tenfoldcv as kfxv
import numpy as np
#import tuning as tu

r.seed(t.time())
lc: list[list] = []
lr: list[list] = []

X = []
Y = []

""" #generate list of n vectors with random x and y values, and the associated class
n = 100
for i in range(n):
  x = r.random()
  y = r.random()
  #classify those items as group 0 if x > y, and group 1 if x <= y
  c = 0 if x > y else 1
  X.append([x, y])
  Y.append(c)

X, Y = kfxv.kfold(X, Y, 10)
Xt = X.pop(0)
Yt = Y.pop(0)
X = kfxv.mergedata(X)
Y = kfxv.mergedata(Y)

classifier = eknn.EKNNErrClassifier()
classifier.fit(np.array(X), np.array(Y))
classifier.edit(Xt, Yt)

classifierp = knn.KNNClassifier()
classifierp.fit(np.array(X), np.array(Y))

print(sum([classifier._predict(x, 5) == y for x, y in zip(Xt, Yt)]))
print(len(classifier.cl.X))
print(sum([classifierp._predict(x, 5) == y for x, y in zip(Xt, Yt)])) """

n = 100
X = np.array([i for i in range(n)])
Y = np.array([(r.random() * np.sqrt(n)) + (i - np.sqrt(n) / 2) for i in range(n)])

X, Y = kfxv.kfold(X, Y, 10)
X_test = X.pop(0)
Y_test = Y.pop(0)
X = kfxv.mergedata(X)
Y = kfxv.mergedata(Y)

X, Y = kfxv.kfold(X, Y, 10)
Xh = X.pop(0)
Yh = Y.pop(0)
X = kfxv.mergedata(X)
Y = kfxv.mergedata(Y)

print(len(X))

regressionE = eknn.EKNNErrRegression()
regressionE.fit(np.array(X), np.array(Y))
regressionE.edit(np.array(X_test), np.array(Y_test), 5, 2)
print(np.mean((regressionE.predict(np.array(Xh), 5, 5) - np.array(Yh)) ** 2))
print(len(regressionE.cl.X))
regressionP = knn.KNNRegression()
regressionP.fit(np.array(X), np.array(Y))
print(np.mean((regressionP.predict(np.array(Xh), 5, 5) - np.array(Yh)) ** 2)) 


""" 
print(np.array([[np.sqrt(np.sum(x - y) ** 2) for y in ys] for x in xs]))
print(np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(xs, ys)]))) """
