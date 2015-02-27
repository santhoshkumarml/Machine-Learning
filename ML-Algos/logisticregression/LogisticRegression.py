'''
Created on Feb 18, 2015

@author: santhosh
'''

import math
import numpy

class LogisticRegression:
    def __init__(self, step_size):
        self.paramVector = []
        self.step_size = step_size    
    
    def calculateDotProductOfVectors(self, paramVector, X):
        arr = numpy.dot(paramVector, X)
        return arr.item(0)
    
    def calculateExponentOfVectors(self, paramVector, X):
        w_x_t = self.calculateDotProductOfVectors(paramVector, X)
        return math.exp(w_x_t)
    
    def getChanged_X_AndTranspose(self, X):
        changed_X = numpy.insert(X, 0, 1)
        changed_X_t = numpy.transpose(changed_X)
        return changed_X, changed_X_t
    
    def calculateHypothesis(self, paramVector, X):
        changed_X,changed_X_t = self.getChanged_X_AndTranspose(X)
        exp_w_x_t = self.calculateExponentOfVectors(paramVector, changed_X_t)
        return 1/(1+exp_w_x_t)
    
    def determine_diff_likehood(self, x, y, exp_w_x_t):
        sigmoid = exp_w_x_t/(1 + exp_w_x_t)
        return x*(y - sigmoid)
    
    def calculateLogLikeliHood(self, X, y, paramVector):
        log_likelihood_list = []
        for i in range(len(y)):
            changed_X,changed_X_t = self.getChanged_X_AndTranspose(X[i])
            w_x_t = self.calculateDotProductOfVectors(paramVector, changed_X_t)
            calc = (y[i]*w_x_t)- math.log(1+math.exp(w_x_t))
            log_likelihood_list.append(calc)
        return  sum(log_likelihood_list)
        
    def fit(self, X, y):
        train_samples,train_features = X.shape
        #no_of_classes = len(set(y))
     
        # estimate parameters
        self.paramVector = numpy.zeros(train_features+1)
        log_likelihood = -float('inf')
        
        #iterative gradient ascend and estimate parameters
        while True:
            paramVector = numpy.zeros(train_features+1)
            diff_l_w = numpy.zeros(train_features+1)
            for sample_idx in range(train_samples):
                changed_X, changed_X_t = self.getChanged_X_AndTranspose(X[sample_idx])
                exp_w_x_t = self.calculateExponentOfVectors(self.paramVector, changed_X_t)
                for param_idx in range(train_features+1):
                    diff_l_w[param_idx]+= self.determine_diff_likehood(changed_X[param_idx],\
                                                                        y[sample_idx], exp_w_x_t)
            for param_idx in range(train_features+1):  
                paramVector[param_idx] = self.paramVector[param_idx] + (self.step_size * diff_l_w[param_idx])
                
            new_log_likelihood = self.calculateLogLikeliHood(X, y, paramVector)

            if new_log_likelihood < log_likelihood or (new_log_likelihood-log_likelihood) < 0.1:
                print 'Final Likelihood', new_log_likelihood, log_likelihood
                break
            else:
                log_likelihood = new_log_likelihood
                self.paramVector = paramVector

    def predict(self, X):
        predicted_value =  self.calculateHypothesis(self.paramVector, X)
        return (0 if predicted_value>=0.5  else 1)