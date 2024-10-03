import copy
import numpy as np
import training as tr
import random

def kfold(D, k, debug = False):
  #copy the vectors in D to a vector list Vs
  Vs = copy.deepcopy(D)
  #shuffle the vectors in Vs
  random.shuffle(Vs)
  #sort the vectors in Vs by their class. Since the sort is stable, the randomization introduced by the shuffle in the previous step will be preserved
  Vs.sort(key=lambda x: int(x[-1]))
  #initialize our list of vector lists Vss, which will represent the folds
  Vss = [[] for _ in range(k)]
  #iterate over the indices for each vector in Vs
  for i in range(len(Vs)):
    '''
    In this loop we use the clock property of modulus arithmetic to create stratification in our data.
    With this method we effectively map data like so:
      Given k = 3 and l = [0, 1, 2, 3, 4, 5]
      stratification(l) = [[0, 3], [1, 4], [2, 5]]
    '''
    Vss[i % k].append(Vs[i])
  #debug code for checking that stratification is working properly
  if debug:
    #iterate over the list of lists of vectors
    for VssVs in Vss:
      #iterate over the vectors in the list of vectors from the list of lists of vectors
      for VssVsV in VssVs:
        #print the vector from the list of vectors from the list of lists of vectors
        print(VssVsV)
      #print a line break to separate each list of vectors from the list of lists of vectors
      print()
  #We return the list of lists produced by the instructions in this function
  return Vss

def mergedata(Ds):
  #initialize our list that represents the merged data
  Dm = []
  #iterate through the chunks in Ds
  for D in Ds:
    #extend the merged data by the vectors in D
    Dm.extend(D)
  #return the vector list that resulted from the execution of this function
  return Dm

def crossvalidationC(D, c, t : tr.Classifier, k=10):
  #first we initialize our confusion matrix list
  cml = [None] * k
  #o is our original list of folds
  o = kfold(D, k)
  for i in range(0, k):
    #initialize the confusion matrix for this fold
    cm = [([0] * len(c)) for _ in range(len(c))]
    #copy the original list to prevent mutating the main list
    fs = copy.copy(o)
    #remove the ith fold from the list and keep it for testing
    tf = fs.pop(i)
    #merge the remaining folds to be used for training
    fs = mergedata(fs)
    #train the classifier on the training data
    t.train(fs) #TODO this is a placeholder for the trainer
    #We classify each vector x in the training fold tf
    for x in tf:
      #we access the confusion matrix position defined by the predicted class and the actual class for x and we increment the value in that position by 1
      cm[t.classify(x)][x[-1]] += 1 #TODO this is a placeholder for the classifier
    #when we are finished with the fold, we save the confusion matrix into our list of confusion matrices
    cml[i] = cm
  #return the list of confusion matrices that were produced with our classifier
  return cml

def crossvalidationR(D, t : tr.Regression, k=10):
  pass