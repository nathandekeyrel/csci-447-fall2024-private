import knn
import heapq as hq
import numpy as np
import copy

class EKNNErrClassifier:
  def __init__(self, D : list[list], T : list[list], k):
    self.D = copy.deepcopy(D)
    self.T = copy.deepcopy(T)
    self.k = k
    self.q = quality(self, T, k)
    self._edit(k)
  
  def _edit(self, k):
    while not self.degrade():
      i = 0
      delta = 0
      while i < len(self.D):
        c = self.classify(self.D[i], k) # I don't know what the k should be in this case
        if not c == self.D[i][-1]:
          delta += 1
          self.D.pop(i)
        i += 1
      if delta == 0:
        return
  
  #take feature vector x, and integer k
  def classify(self, x, k):
    #initialize the list we will use for our k nearest neighbors
    l = []
    #for each vector xt in D, measure the distance between x and xt
    for xt in self.D:
      #save the distance in a tuple named e with its associated class
      e = (-knn.euclidianDistance(x, xt), xt[-1])
      #we used the heappushpop function since the documentation claims it is more efficient than using the two functions separately
      if len(l) >= k:
        hq.heappushpop(l, e)
      else:
        hq.heappush(l, e)
    #initialize the dictionary for tallying
    v = {}
    #our for loop for tallying the votes
    for ct in l:
      #if the class is present then we only need to increment the value by one
      if ct[1] in v:
        v[ct[1]] += 1
      #otherwise we have to create an entry for that value, and initialize it as 1
      else:
        v.update({ct[1] : 1})
    #we convert the dictionary to a list of tuples
    v = v.items()
    #sort the list in reverse by the number of votes, so that the item with the highest votes wins
    v = sorted(v, key=lambda item: item[1], reverse=True)
    #return the class of the first item in the list
    return v[0][0]
  
  def degrade(self):
    if quality(self, self.T, self.k) < self.q:
      return True
    return False

class EKNNTrueClassifier:
  def __init__(self, D : list[list], T : list[list], k):
    self.D = copy.deepcopy(D)
    self.T = copy.deepcopy(T)
    self.k = k
    self.q = quality(self, T, k)
    self._edit(k)
  
  def _edit(self, k):
    while not self.improve():
      i = 0
      delta = 0
      while i < len(self.D):
        c = self.classify(self.D[i], k)
        if c == self.D[i][-1]:
          delta += 1
          self.D.pop(i)
        i += 1
      if delta == 0:
        return
  
  #take feature vector x, and integer k
  def classify(self, x, k):
    #initialize the list we will use for our k nearest neighbors
    l = []
    #for each vector xt in D, measure the distance between x and xt
    for xt in self.D:
      #save the distance in a tuple named e with its associated class
      e = (-knn.euclidianDistance(x, xt), xt[-1])
      #we used the heappushpop function since the documentation claims it is more efficient than using the two functions separately
      if len(l) >= k:
        hq.heappushpop(l, e)
      else:
        hq.heappush(l, e)
    #initialize the dictionary for tallying
    v = {}
    #our for loop for tallying the votes
    for ct in l:
      #if the class is present then we only need to increment the value by one
      if ct[1] in v:
        v[ct[1]] += 1
      #otherwise we have to create an entry for that value, and initialize it as 1
      else:
        v.update({ct[1] : 1})
    #we convert the dictionary to a list of tuples
    v = v.items()
    #sort the list in reverse by the number of votes, so that the item with the highest votes wins
    v = sorted(v, key=lambda item: item[1], reverse=True)
    #return the class of the first item in the list
    return v[0][0]
  
  def improve(self):
    if not (quality(self, self.T, self.k) > self.q):
      return True
    return False

class EKNNErrRegression:
  def __init__(self, D : list[list], T : list[list], k, e):
    self.D = copy.deepcopy(D)
    self.T = copy.deepcopy(T)
    self.k = k
    self.e = e
    self.q = qualityR(self, T, k)
    self._edit(k)
  
  def _edit(self, k):
    while not self.degrade():
      i = 0
      delta = 0
      while i < len(self.D):
        p = self.predict(self.D[i], k)
        if not (self.D[i][-1] - self.e <= p <= self.D[i][-1] + self.e):
          delta += 1
          self.D.pop(i)
        i += 1
      if delta == 0:
        return
  
  #take feature vector x, and integer k
  def predict(self, x, k):
    #initialize the list we will use for our k nearest neighbors
    l = []
    #for each vector xt in D, measure the distance between x and xt
    for xt in self.D:
      #save the distance in a tuple named e with its associated class
      e = (-knn.euclidianDistance(x, xt), xt[-1])
      #we used the heappushpop function since the documentation claims it is more efficient than using the two functions separately
      if len(l) >= k:
        hq.heappushpop(l, e)
      else:
        hq.heappush(l, e)
    #get the average of the target values of the k nearest neighbors
    return sum(e[1] for e in l) / k
  
  def degrade(self):
    if qualityR(self, self.T, self.k) < self.q:
      return True
    return False

class EKNNTrueRegression:
  def __init__(self, D : list[list], T : list[list], k, e):
    self.D = copy.deepcopy(D)
    self.T = copy.deepcopy(T)
    self.k = k
    self.e = e
    self.q = qualityR(self, T, k)
    self._edit(k)
  
  def _edit(self, k):
    while not self.improve():
      i = 0
      delta = 0
      while i < len(self.D):
        p = self.predict(self.D[i], k)
        if (self.D[i][-1] - self.e <= p <= self.D[i][-1] + self.e):
          delta += 1
          self.D.pop(i)
        i += 1
      if delta == 0:
        return
  
  #take feature vector x, and integer k
  def predict(self, x, k):
    #initialize the list we will use for our k nearest neighbors
    l = []
    #for each vector xt in D, measure the distance between x and xt
    for xt in self.D:
      #save the distance in a tuple named e with its associated class
      el = (-knn.euclidianDistance(x, xt), xt[-1])
      #we used the heappushpop function since the documentation claims it is more efficient than using the two functions separately
      if len(l) >= k:
        hq.heappushpop(l, el)
      else:
        hq.heappush(l, el)
    #get the average of the target values of the k nearest neighbors
    return sum(el[1] for el in l) / k
  
  def improve(self):
    if not (qualityR(self, self.T, self.k) > self.q):
      return True
    return False

def quality(C, T, k):
  n = 0
  d = 0
  for t in T:
    d += 1
    if C.classify(t, k) == t[-1]:
      n += 1
  try:
    r = n / d
  except:
    r = 1
  return r

def qualityR(R, T, k):
  e = 0
  for t in T:
    te = R.predict(t, k)
    e += (te * te)
  return e / len (T)