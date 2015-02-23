'''
Created on Feb 18, 2015

@author: santhosh
'''

import math
import numpy
import sys

class LogisticRegression:
    def __init__(self, step_size):
        self.paramVector = []
        self.step_size = step_size    
    
    def calculateDotProductOfVectors(self, paramVector, X):
        return numpy.dot(paramVector, X)
    
    def calculateExponentOfVectors(self, paramVector, X):
        return math.exp(self.calculateDotProductOfVectors(paramVector, X))
       
    def sigmoid(self,paramVector, X):
        calc = self.calculateExponentOfVectors(paramVector, X)
        return calc/(1+calc)
    
    def getChange_X(self, X):
        changed_X = numpy.insert(X, 0, 1)
        return changed_X
    
    def getChange_X_AndTranspose(self, X):
        changed_X = self.getChange_X(X)
        changed_X_t = numpy.transpose(changed_X)
        return changed_X_t
    
    def calculateHypothesis(self, paramVector, X):
        changed_X_t = self.getChange_X_AndTranspose(X)    
        return self.sigmoid(self.paramVector, changed_X_t)
    
    def determine_step_factor(self, j, paramVector, X, y):
        changed_X = numpy.insert(X, 0, 1)
        changed_X_t = numpy.transpose(changed_X)
        return X[j]*(y - self.calculateExponentOfVectors(paramVector, changed_X_t))
    
    def calculateLogLikeliHood(self, train_samples, X, y, paramVector):
        log_likelihood_list = []
        for i in range(train_samples):
            changed_X = numpy.insert(X[i], 0, 1)
            changed_X_t = numpy.transpose(changed_X)
            calc = y[i]*(self.calculateDotProductOfVectors(paramVector, changed_X_t))-\
             math.log(1+self.calculateExponentOfVectors(paramVector, changed_X_t))
            log_likelihood_list.append(calc)
            
        return  sum(log_likelihood_list)
        
    def fit(self, X, y):
        train_samples,train_features = X.shape
        #no_of_classes = len(set(y))
     
        # estimate parameters
        self.paramVector = numpy.zeros(train_features+1)
        log_likelihood = self.calculateLogLikeliHood(train_samples, X, y, self.paramVector)
        
        #iterative gradient ascend and estimate parameters
        while True:
            paramVector = numpy.zeros(train_features+1)
            for j in range(train_features+1):
                step_factor = 0
                for i in range(train_samples):
                    step_factor += self.determine_step_factor(j, self.paramVector,\
                                                  X[i], y[i])
                paramVector[j] = self.paramVector[j] + (self.step_size * step_factor)
                
            new_log_likelihood = self.calculateLogLikeliHood(train_samples, X, y, paramVector)
            if new_log_likelihood < log_likelihood:
                break
            else:
                log_likelihood = new_log_likelihood
                self.paramVector = paramVector
                
        for i in range(train_features+1):
            print self.paramVector[i]
    
    def predict(self, X):
        predicted_value =  self.calculateHypothesis(self.paramVector, X)
        return (0 if predicted_value >=0.5  else 1)