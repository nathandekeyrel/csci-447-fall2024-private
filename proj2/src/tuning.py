import tenfoldcv as kfxv
import numpy as np
import knn
import copy

NUMGUESSES = 10

def tuneKNN(X, c):
  '''
  : param X : the processed data
  : param c : the list of classes associated with that data
  '''
  # make ten folds
  X = kfxv.kfold(X, 10)
  # set aside one fold for tuning
  tuningSet = X.pop(0)
  # merge nine of them
  X = kfxv.mergedata(X)
  # make ten folds from the merged data
  X = kfxv.kfold(X, 10)
  # For the KNN classifier, there is one hyperpameter which we will initialize as an empty list for averaging later
  kns = []
  # tune the hyper parameters
  for i in range(10):
    # make a copy to preserve X
    Xc = copy.deepcopy(X)
    # We remove the fold at index and use it for our hold-out
    h = Xc.pop(i)
    #merge the data
    Xc = kfxv.mergedata(Xc)
    # We'll generate a list of kn values to test against for this iteration
    knt = np.random.randint(0, int(np.sqrt(len(Xc))), NUMGUESSES)
    # initialize the classifier
    classifier = knn.KNNClassifier(Xc)
    # initialize an array to store the 0-1 results for each kn value in knt
    zeroones = []
    # iterate each kn value in the knt list
    for kn in knt:
      # check the performance of the kn
      for x in h:
        zeroone = 0
        # store the predicted class into c
        c = classifier.classify(x, kn)
        # compare predicted class to the actual and increment the results based on the outcome
        zeroone += 1 if not c == x[-1] else 0
      # once loop is done, append the zero one results to the list
      zeroones.append(zeroone / len(h))
    # append the kns list with the best performing kn value for the training set
    kns.append(knt[zeroones.index(max(zeroones))])
  return np.mean(kns)