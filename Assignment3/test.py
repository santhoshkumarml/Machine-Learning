__author__ = 'santhosh'

import scipy.io
import os
from knn import KNN
import numpy

def measureAccuracyOnTestData(knnClassifier):
    global accuracy, i, label
    accuracy = 0.0
    for i in range(len(test_data)):
        label = knnClassifier.predict(test_data[i])
        if label == test_result[i]:
            accuracy += 1.0
    accuracy /= len(test_data)
    print accuracy


def extractData():
    matlab_data = scipy.io.loadmat(os.path.join(os.getcwd(), 'resources/faces.mat'))
    train_data = matlab_data['traindata']
    train_result = numpy.array([labels[0] for labels in matlab_data['trainlabels']])
    test_data = matlab_data['testdata']
    test_result = numpy.array([labels[0] for labels in matlab_data['testlabels']])
    return train_data, train_result, test_data, test_result

train_data, train_result, test_data, test_result = extractData()
knnClassifier = KNN(range(1,100), 10)
knnClassifier.fit(train_data, train_result)
measureAccuracyOnTestData(knnClassifier)
