__author__ = 'santhosh'

import scipy.io
import os
from knnclassifier import KNNClassifier
import numpy
from datetime import datetime


def extractData():
    matlab_data = scipy.io.loadmat(os.path.join(os.getcwd(), 'resources/faces.mat'))
    train_data = matlab_data['traindata']
    train_result = numpy.array([labels[0] for labels in matlab_data['trainlabels']])
    test_data = matlab_data['testdata']
    test_result = numpy.array([labels[0] for labels in matlab_data['testlabels']])
    return train_data, train_result, test_data, test_result

print '----------------------------------------------------------------------------------------'
beforeTime = datetime.now()
train_data, train_result, test_data, test_result = extractData()
knnClassifier = KNNClassifier(range(1,100), 10)
knnClassifier.fitPredictAndScore(train_data, train_result, test_data, test_result)
afterTime = datetime.now()
print 'Time Taken', afterTime - beforeTime
print '----------------------------------------------------------------------------------------'