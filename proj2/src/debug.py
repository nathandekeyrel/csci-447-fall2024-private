  import knn as knn
import editedKNN as eknn
import random as r
import time as t
import tenfoldcv as kfxv
import numpy as np

r.seed(t.time())
lc : list[list] = []
lr : list[list] = []

#generate list of n vectors with random x and y values, and the associated class
n = 1000
for i in range(n):
  x = r.random()
  y = r.random()
  #classify those items as group 0 if x > y, and group 1 if x <= y
  c = 0 if x > y else 1
  lc.append([x, y, c])

n = 100
for i in range(n):
  x = i
  y = (r.random() * np.sqrt(n)) + (i - np.sqrt(n) / 2)
  lr.append([x, y])

lr = kfxv.kfold(lr, 10)
trr = lr.pop(0)
lr = kfxv.mergedata(lr)

regression = eknn.EKNNTrueRegression(lr, trr, 5, 5)

for x in trr:
  print(str(x[0]), " ", str(regression.predict(x, 5, )))








'''
lc = kfxv.kfold(lc, 10)
trc = lc.pop(0)
lc = kfxv.mergedata(lc)
classifier = eknn.EKNNErrClassifier(lc, trc, 5)
classifier2 = eknn.EKNNTrueClassifier(lc, trc, 5)
'''