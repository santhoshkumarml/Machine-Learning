'''
Created on Feb 18, 2015

@author: santhosh
'''

import numpy


class NaiveBayesClassifier:
 def __init__(self):
     self.cpt = []
     pass
 
 def addOneSmoothing(self):
     pass
 
 def fit(self, X, y):
     k = set(y)
     d = X.shape[0]
     
     self.cpt = numpy.zeros(shape = (),dtype = float32)
 
 def predict(self, X):
     pass