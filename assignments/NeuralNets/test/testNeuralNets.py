__author__ = 'santhosh'
import math
import numpy


# the idea here is to make the input instance with 1 as the first feature and transpose it
def getChanged_X_AndTranspose(X):
    changed_X = numpy.insert(X, 0, 1)
    changed_X_t = numpy.transpose(changed_X)
    return changed_X, changed_X_t

# calcuate W_Trapose * X (instead of w(0)+ sum(w(i)*x(i) for all i>=1)
# x(0) is always 1
# we will do sum(w(i)*x(i)) for all i>=
def calculateDotProductOfVectors(paramVector, X):
    arr = numpy.dot(paramVector, X)
    return arr.item(0)


# calculate math.exp(W_Trapose * X)
def calculateExponentOfVectors(paramVector, X):
    w_x_t = calculateDotProductOfVectors(paramVector, X)
    return math.exp(w_x_t)


# Calculate P(y=0/x)
def calculateSigmoid(paramVector, X):
    changed_X, changed_X_t = getChanged_X_AndTranspose(X)
    exp_w_x_t = calculateExponentOfVectors(paramVector, changed_X_t)
    return 1 / (1 + exp_w_x_t)
