import knn
import heapq as hq
import numpy as np
import copy

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
    return e / len(T)
