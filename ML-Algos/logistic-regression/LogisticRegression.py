'''
Created on Feb 18, 2015

@author: santhosh
'''
import sys
import numpy

class LogisticRegression:
    def __init__(self, step_size):
        self.paramVector = []
        self.step_size = step_size    
        
    def calculateExponentOfVectors(self, paramVector, X):
        return math.exp(-numpy.dot(paramVector, X))
       
    def sigmoid(self,paramVector, X):
        return  1/(1 + self.calculateExponentOfVectors(paramVector, X))
    
    def calculateHypothesis(self, paramVector, X):
        changed_X = numpy.insert(X, 0, 1)
        changed_X_t = numpy.transpose(changed_X)
        return
    
    def determine_step_factor(self, j, paramVector, X, y):
        return X[j]*(y - self.calculateExponentOfVectors(paramVector, X))
    
    def fit(self, X, y):
        train_samples,train_features = X.shape
        #no_of_classes = len(set(y))
     
        # estimate parameters
        self.paramVector = numpy.zeros(shape = (1, train_features+1))
        log_likelihood = sum([(y[i]-self.calculateHypothesis(X[i])) for i in range(train_samples)])
     
        #iterative gradient ascend
        while True:
            paramVector = numpy.zeros(shape = (1, train_features+1))
            
            
            for j in range(len(self.paramVector)):
                paramVector[j] = self.paramVector
                step_factor = 0
                for i in range(train_samples):
                    step_factor += self.determine_step_factor(j, self.paramVector,\
                                                  X[i], y[i])
                paramVector[j] = self.paramVector+ (self.step_size * step_factor)
                
            new_log_likelihood = sum([(y[i]-self.calculateHypothesis(paramVector, X[i]))\
                                       for i in range(train_samples)])
            
            if new_log_likelihood < log_likelihood:
                break
            else:
                log_likelihood = new_log_likelihood
                self.paramVector = paramVector
    
    def predict(self, X):
        no_of_features = X.shape[1]
        predicted_value =  self.calculateHypothesis(X)
        return (0 if predicted_value >=0.5  else 1)