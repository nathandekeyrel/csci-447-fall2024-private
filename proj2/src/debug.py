import knn as knn
import editedKNN as eknn
import kMeans as km
import random as r
import time as t
import tenfoldcv as kfxv
import numpy as np
import tuning as tu
import preprocess as prpr
import evaluating as ev

n = 100

r.seed(t.time())
lc: list[list] = []
lr: list[list] = []

X = []
Y = []

""" X, Y = prpr.preprocess_data("data/breast-cancer-wisconsin.data")
Xs, Ys, X_test, Y_test = tu.generateStartingTestData(X, Y)
Xs = kfxv.mergedata(Xs)
Ys = kfxv.mergedata(Ys)
cl = km.KMeans(100)
cl.fit(Xs, Ys)
print(cl.predict(X_test))
print() """

""" X, Y = prpr.preprocess_data("data/breast-cancer-wisconsin.data")

k = tu.tuneKNNClassifier(X, Y)
print(k)
cl = knn.KNNClassifier()
cms = kfxv.tenfoldcrossvalidationC(cl, X, Y, k)
print(cms) """

""" X, Y = prpr.preprocess_data("data/soybean-small.data")

k = tu.tuneKNNClassifier(X, Y)
print(k) """

""" X, Y = prpr.preprocess_data("data/breast-cancer-wisconsin.data")
cl = knn.KNNClassifier()
cms = kfxv.tenfoldcrossvalidationC(cl, X, Y, 5)
print(cms) """

""" X, Y = prpr.preprocess_data("data/breast-cancer-wisconsin.data")
print(len(Y))
Xs, Ys = kfxv.kfold(X, Y, 10)
Xh = Xs.pop(0)
Yh = Ys.pop(0)
Xs = kfxv.mergedata(Xs)
Ys = kfxv.mergedata(Ys)

time1 = t.time()
# re = knn.KNNRegression()
# re = eknn.EKNNErrRegression()
re = eknn.EKNNErrClassifier()
re.fit(Xs, Ys)
time2 = t.time()
print(str(time2 - time1))
time2 = t.time()
# re.edit(Xh, Yh, 1, (max(Y) - min(Y)) * 0.25)
re.edit(Xh, Yh)
time3 = t.time()
print(str(time3 - time2))
time3 = t.time()
# pred = re.predict(Xh, 13, 0.25)
pred = re.predict(Xh, 5)
time4 = t.time()
print(str(time4 - time3))
time4 = t.time()
# results = kfxv.tenfoldcrossvalidationR(re, X, Y, 5, 2, e=5)
time5 = t.time()
print(str(time5 - time4))
print(re.mark.sum())
# print(tu.r2_score(Yh, pred))
print(ev.zero_one_loss(np.array(Yh), np.array(pred))) """

""" X, Y = prpr.preprocess_data("data/machine.data")
tu.tuneEKNNRegression(X, Y)
tu.tuneKNNRegression(X, Y)
tu.tuneKMeansRegression(X, Y, 40)

X, Y = prpr.preprocess_data("data/soybean-small.data")
tu.tuneEKNNClassifier(X, Y)
tu.tuneKNNRegression(X, Y)
tu.tuneKMeansClassifier(X, Y, 10) """


tu.tuneEverything("data/")
# X, Y = prpr.preprocess_data("data/machine.data")
# k, sig, e, kc = tu.tuneEKNNRegression(X, Y)
# print(kc)

""" X, Y = prpr.preprocess_data("data/abalone.data")

time1 = t.time()
k, sig, e = tu.tuneEKNNRegression(X, Y)
time2 = t.time()
print(time2 - time1)
print(k, sig)
Xs, Ys, X_test, Y_test = tu.generateStartingTestData(X, Y)
Xs = kfxv.mergedata(Xs)
Ys = kfxv.mergedata(Ys)
re = eknn.EKNNErrRegression()
re.fit(Xs, Ys)
re.edit(X_test, Y_test, sig, e)
y_pred = re.predict(X_test, k, sig)
r2 = tu.r2_score(Y_test, y_pred)
mse = ev.mse(Y_test, y_pred)
print(r2)
print(mse) """

""" X = np.array([i for i in range(n)])
Y = np.array([(r.random() * np.sqrt(n)) + (i - np.sqrt(n) / 2) for i in range(n)])

Xs, Ys = kfxv.kfold(X, Y, 10)
Xh = np.array(Xs.pop(0))
Yh = np.array(Ys.pop(0))
Xt = np.array(kfxv.mergedata(Xs))
Yt = np.array(kfxv.mergedata(Ys))

classifier = km.KMeans(2, 100)
classifier.fit(Xt, Yt)
results = classifier.predict(Xh)
# results = classifier.get_reduced_dataset()
print((Yh, results)) """

""" n = 100
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
print(np.mean((regressionP.predict(np.array(Xh), 5, 5) - np.array(Yh)) ** 2))  """


""" 
print(np.array([[np.sqrt(np.sum(x - y) ** 2) for y in ys] for x in xs]))
print(np.sqrt(np.sum([(x - y) ** 2 for x, y in zip(xs, ys)]))) """


""" paths = ["data/abalone.data", "data/breast-cancer-wisconsin.data", "data/forestfires.csv", "data/glass.data", "data/machine.data", "data/soybean-small.data"]
for path in paths:
  X, Y = prpr.preprocess_data(path) """
  