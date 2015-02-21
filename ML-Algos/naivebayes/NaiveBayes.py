'''
Created on Feb 18, 2015

@author: santhosh
'''

import numpy

class NaiveBayesClassifier:
 def __init__(self):
     self.jpd = []
     self.fp = []
     self.tp = []
     pass
 
 
 def fit(self, X, y):
     train_samples,train_features = X.shape
     no_of_classes = len(set(y))
     
     total_words = sum([sum(X[i]) for i in range(train_samples)])
     
     self.cpt = numpy.zeros(shape = (train_features, no_of_classes),dtype = float32)
     self.tp = numpy.zeros(no_of_classes)
     self.fp = numpy.zeros(train_features)
     
     for i in range(train_samples):
         classified_class = y[i]
         self.tp[classified_class] += 1
         for j in range(train_features):
             #total_count_of_jth_feature = sum(X[:,j])
             self.cpt[j][classified_class] += X[i][j]
             self.fp[j]+=1
            
     for i in range(train_features):
         total_count_of_ith_feature = sum(self.cpt[i])
         self.fp[i] /= total_words
         for j in range(no_of_classes):
             self.cpt[i][j] += 1
             self.cpt[i][j] /= (total_count_of_ith_feature + total_words) 

 def predict(self, X):
     pass