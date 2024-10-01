import knn as knn
import random as r
import time as t

r.seed(t.time())
l : list[list]= []
#generate list of twenty vectors with random x and y values, and the associated class
for i in range(100):
  x = r.random()
  y = r.random()
  #classify those items as group 0 if x > y, and group 1 if x <= y
  c = 0 if x > y else 1
  l.append([x, y, c])

#generate test set
lt : list[list] = []
for i in range(5):
  x = r.random()
  y = r.random()
  c = 0 if x > y else 1
  lt.append([x, y, c])

cl = knn.KNNClassifier(l)
for x in lt:
  cc = cl.classify(x, 5)
  c = x[-1]
  print(str(cc) + " " + str(c))