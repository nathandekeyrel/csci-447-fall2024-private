import numpy as np
import heapq as hq

class KNNClassifier:
  #init with the data D
  def __init__(self, D : list):
    self.D = D
  
  #take feature vector x, and integer k
  def classify(self, x, k):
    #initialize the list we will use for our k nearest neighbors
    l = []
    #for each vector xt in D, measure the distance between x and xt
    for xt in self.D:
      #save the distance in a tuple named e with its associated class
      ct = (-euclidianDistance(x, xt), xt[-1])
      #we used the heappushpop function since the documentation claims it is more efficient than using the two functions separately
      if len(l) >= k:
        hq.heappushpop(l, ct)
      else:
        hq.heappush(l, ct)
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

class KNNRegression:
  #init with the data D
  def __init__(self, D : list):
    self.D = D
  
  #take feature vector x, and integer k
  def predict(self, x, k):
    #initialize the list we will use for our k nearest neighbors
    l = []
    #for each vector xt in D, measure the distance between x and xt
    for xt in self.D:
      #save the distance in a tuple named e with its associated class
      e = (-euclidianDistance(x, xt), xt[-1])
      #we used the heappushpop function since the documentation claims it is more efficient than using the two functions separately
      if len(l) >= k:
        hq.heappushpop(l, e)
      else:
        hq.heappush(l, e)
    #get the average of the target values of the k nearest neighbors
    return sum(e[1] for e in l) / k

#euclidian distance algorithm
def euclidianDistance(x1, x2):
  #initialize distance to zero, since we haven't measured anything yet
  distance = 0.0
  #if the lengths aren't equal, return a -1, we should replace this with thrwoing an error
  if not len(x1) == len(x2):
    return -1.0
  #find the Euclidean distance between two vectors
  for i in range(len(x1) - 1):
    #first we will use a temporary variable to get the difference between the units at position i in the vector x1 and the vector x2
    d = x1[i] - x2[i]
    #then we add the square of that difference to the distance value
    distance += (d * d)
  #after the squares of the differences are all added up, we get the square root of the final distance value to get the actual Euclidean distance
  distance = np.sqrt(distance)
  #then we return that ditance to close out the frame
  return distance