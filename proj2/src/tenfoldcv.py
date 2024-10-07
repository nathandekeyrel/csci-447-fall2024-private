import copy
import numpy as np
import training as tr
import random
import knn
import editedKNN as eknn

def kfold(X, Y, k, debug = False):
    #copy the vectors in D to a vector list Vs
    Vs = list(zip(X, Y))
    #shuffle the vectors in Vs
    random.shuffle(Vs)
    #sort the vectors in Vs by their class. Since the sort is stable, the randomization introduced by the shuffle in the previous step will be preserved
    Vs.sort(key=lambda x: x[-1])
    #initialize our list of vector lists Vss, which will represent the folds
    Xs = [[] for _ in range(k)]
    Ys = [[] for _ in range(k)]
    #iterate over the indices for each vector in Vs
    for i in range(len(Vs)):
        '''
        In this loop we use the clock property of modulus arithmetic to create stratification in our data.
        With this method we effectively map data like so:
          Given k = 3 and l = [0, 1, 2, 3, 4, 5]
          stratification(l) = [[0, 3], [1, 4], [2, 5]]
        '''
        Xs[i % k].append(Vs[i][0])
        Ys[i % k].append(Vs[i][1])
    #We return the list of lists produced by the instructions in this function
    return Xs, Ys

def _crossvalidationC(i, X, Y, nClasses, k, cl):
    # copy the X and Y arrays to keep the sample data intact
    Xc = copy.copy(X)
    Yc = copy.copy(Y)
    # pop the ith element for the holdout set
    Xh = np.array(Xc.pop(i))
    Yh = np.array(Yc.pop(i))
    # merge the remaining data for the training set
    Xt = np.array(mergedata(Xc))
    Yt = np.array(mergedata(Yc))
    # fit the model to the training data
    cl.fit(Xt, Yt)
    # if there is an edit method, run that using the hold out set
    if hasattr(cl, 'edit') and callable(cl.edit):
        cl.edit(Xh, Yh)
    # get the predictions
    predictions = cl.predict(Xh, k)
    # initialize the confusion matrix
    cm = np.array([[0 for _ in range(nClasses)] for _ in range(nClasses)])
    # generate the confusion matrix
    for x, y in zip(predictions, Yh):
        cm[x][y] += 1
    # return the confusion matrix
    return cm

def tenfoldcrossvalidationC(cl, X, Y, k):
    nClasses = np.max(Y) + 1
    X, Y = kfold(X, Y, 10)
    cms = [_crossvalidationC(i, X, Y, nClasses, k, cl) for i in range(10)]
    return cms

def _crossvalidationR(i, X, Y, sig, k, re, e=0):
    # copy the X and Y arrays to keep the sample data intact
    Xc = copy.copy(X)
    Yc = copy.copy(Y)
    # pop the ith element for the holdout set
    Xh = np.array(Xc.pop(i))
    Yh = np.array(Yc.pop(i))
    # merge the remaining data for the training set
    Xt = np.array(mergedata(Xc))
    Yt = np.array(mergedata(Yc))
    # fit the model to the training data
    re.fit(Xt, Yt)
    # if there is an edit method, run that using the hold out set
    if hasattr(re, 'edit') and callable(re.edit):
        re.edit(Xh, Yh, sig, e)
    # get the predictions
    predictions = re.predict(Xh, k, sig)
    # return a tuple containing the hold out target values and the predictions
    return (copy.copy(Yh), predictions)

def tenfoldcrossvalidationR(re, X, Y, k, sig, e=0):
    X, Y = kfold(X, Y, 10)
    results = [_crossvalidationR(i, X, Y, sig, k, re, e) for i in range(10)]
    return results

def mergedata(Ds):
    # initialize our list that represents the merged data
    Dm = []
    # iterate through the chunks in Ds
    for D in Ds:
        # extend the merged data by the vectors in D
        Dm.extend(D)
    # return the vector list that resulted from the execution of this function
    return Dm
