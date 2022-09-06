import csv
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import itertools
import time

from scipy.spatial import procrustes
from numpy.linalg import svd
from numpy.linalg import norm
import scipy.linalg as la

'''
The three metrics used in the analysis. 
Caution that the functions return d^2
'''

def frobenius(X,Y):
    return norm(X-Y)**2

def square_root(X,Y):
    d,v = la.eigh(X)
    d = np.maximum(d,0)
    sqX = np.dot(v,np.dot(np.diag(np.sqrt(d)),v.T))

    d,v = la.eigh(Y)
    d = np.maximum(d,0)
    sqY = np.dot(v,np.dot(np.diag(np.sqrt(d)),v.T))
    return norm(sqX - sqY)**2

def procrustes(X,Y):
    d,v = la.eigh(X)
    d = np.maximum(d,0)
    sqX = np.dot(v,np.dot(np.diag(np.sqrt(d)),v.T))

    d,v = la.eigh(Y)
    d = np.maximum(d,0)
    sqY = np.dot(v,np.dot(np.diag(np.sqrt(d)),v.T))

    R,s = la.orthogonal_procrustes(sqX,sqY)
    return norm(np.dot(sqX,R)-sqY)**2