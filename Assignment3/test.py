__author__ = 'santhosh'

import scipy.io
import os
from knn import KNN
import numpy

matlab_data = scipy.io.loadmat(os.path.join(os.getcwd(),'resources/faces.mat'))
train_data = matlab_data['traindata']
train_result = numpy.array([labels[0] for labels in matlab_data['trainlabels']])
test_data = matlab_data['testdata']
test_result = numpy.array([labels[0] for labels in matlab_data['testlabels']])
knnClassifier = KNN(range(1,100), 10)
knnClassifier.fit(train_data,train_result)