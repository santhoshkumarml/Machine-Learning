'''
Created on Feb 18, 2015

@author: santhosh
'''
import numpy
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.jpd = [] #p(x|y)
        self.fp = [] #p(x)
        self.tp = [] #p(y)
 
 
    def fit(self, X, y):
        train_samples,train_features = X.shape
        no_of_classes = len(set(y))
     
        total_words = sum([sum(X[i]) for i in range(train_samples)])
     
        self.cpt = numpy.zeros(shape = (train_features, no_of_classes))
        self.tp = numpy.zeros(no_of_classes)
        self.fp = numpy.zeros(train_features)
        
        for i in range(train_samples):
            classified_class = y[i]
            self.tp[classified_class] += 1
            for j in range(train_features):
                self.cpt[j][classified_class] += X[i][j]
                self.fp[j]+=1
        
        total_instances = sum(self.tp)
        
        for j in range(no_of_classes):
            self.tp[j] /= total_instances
        
        for i in range(train_features):
            total_count_of_ith_feature = sum(self.cpt[i])
            self.fp[i] /= total_words
            for j in range(no_of_classes):
                self.cpt[i][j] += 1 #add-one smoothing
                self.cpt[i][j] /= (total_count_of_ith_feature + total_words)
        

    def predict(self, X):
        probability = {i:0 for i in range(len(self.tp))}
        for i in range(len(X)):
            for j in range(len(self.tp)):
                calc = ((self.cpt[i][j]**X[i])*self.tp[j])/self.fp[i]
                probability[j] += math.log(calc, 10)
        return max(probability.iterkeys(), key = lambda x: probability[x])