'''
Created on Feb 18, 2015

@author: santhosh
'''
import numpy
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.cpt = [] #p(x|y)
        self.fp = [] #p(x)
        self.tp = [] #p(y)
 
 
    def fit(self, X, y):
        train_samples,train_features = X.shape
        no_of_classes = len(set(y))
     
        total_words = sum([sum(X[i]) for i in range(train_samples)])
     
        self.cpt = numpy.ones(shape = (train_features, no_of_classes)) #add - one smoothing
        self.tp = numpy.zeros(no_of_classes)
        self.fp = numpy.zeros(train_features)
        
        for i in range(train_samples):
            classified_class = y[i]
            self.tp[classified_class] += 1
            for j in range(train_features):
                self.cpt[j][classified_class] += X[i][j]
                self.fp[j] += X[i][j]
        
        
        for j in range(no_of_classes):
            self.tp[j] /= train_samples
            
        total_words_with_add_one_smoothing =  [sum(self.cpt[:,j]) for j in range(no_of_classes)]
        
        for i in range(train_features):
            self.fp[i] /= total_words
            for j in range(no_of_classes):
                self.cpt[i][j] /= total_words_with_add_one_smoothing[j]

    def predict(self, X):
        probability = {i:0 for i in range(len(self.tp))}
        
        for i in range(len(X)):
            for j in range(len(self.tp)):
                    # math overflow error was happening if i simply muliply and power the probablities
                    probability[j] += X[i]*math.log(self.cpt[i][j]) 
                    
        for j in range(len(self.tp)):
            # multiplying p(y)
            probability[j] = probability[j]+math.log(self.tp[j])
                
        return max(probability.iterkeys(), key = lambda key: probability[key])