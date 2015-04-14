__author__ = 'santhosh'

import scipy.io
import os
from knnclassifier import KNNClassifier
import numpy
from datetime import datetime
import random
from sklearn import svm
import math
import util


def extractData():
    matlab_data = scipy.io.loadmat(os.path.join(os.getcwd(), 'resources/faces.mat'))
    train_data = matlab_data['traindata']
    train_result = numpy.array([labels[0] for labels in matlab_data['trainlabels']])
    test_data = matlab_data['testdata']
    test_result = numpy.array([labels[0] for labels in matlab_data['testlabels']])
    return train_data, train_result, test_data, test_result

print '----------------------------------------------------------------------------------------'
print 'Question 1'
beforeTime = datetime.now()
train_data, train_result, test_data, test_result = extractData()
knnClassifier = KNNClassifier(range(1,100), 10)
knnClassifier.fitPredictAndScore(train_data, train_result, test_data, test_result)
afterTime = datetime.now()
print 'Time Taken', afterTime - beforeTime
print '----------------------------------------------------------------------------------------'
print 'Question 2'
beforeTime = datetime.now()
def doNFoldCrossValidation(n, train_data, train_result, c=1):
    n_samples, n_features = train_data.shape
    partition_size = n_samples/n
    test_idx_for_each_iter = [set() for fold in range(n)]
    random_idxs = range(n_samples)
    random_idxs = random.sample(xrange(n_samples), n_samples)
    itr = 0
    for i in range(0, n_samples):
        if len(test_idx_for_each_iter[itr]) != partition_size:
            test_idx_for_each_iter[itr].add(random_idxs[i])
        else:
            itr+=1
    cross_validation_error = 0.0
    train_error = 0.0
    svm_classifier = svm.LinearSVC(C=c)
    svm_classifier_for_train_error = svm.LinearSVC(C=c)
    svm_classifier_for_train_error.fit(train_data, train_result)
    for sample_idx in range(n_samples):
        train_ins = train_data[sample_idx]
        if svm_classifier_for_train_error.predict(train_ins) != train_result[sample_idx]:
            train_error += 1.0
    for i in range(len(test_idx_for_each_iter)):
        test_idxs = test_idx_for_each_iter[i]
        train_data_for_iter, train_result_for_iter = [], []
        for sample_idx in range(n_samples):
            if sample_idx not in test_idxs:
                train_data_for_iter.append(train_data[sample_idx])
                train_result_for_iter.append(train_result[sample_idx])
        svm_classifier.fit(train_data_for_iter, train_result_for_iter)
        for test_idx in test_idxs:
            test_label = train_result[test_idx]
            label = svm_classifier.predict(train_data[test_idx])
            if label != test_label:
                cross_validation_error += 1
    cross_validation_error /= n_samples
    train_error /= n_samples
    return cross_validation_error, train_error
print '----------------------------PART 1 Test Error With C= 500----------------------------------'
svm_classifier = svm.LinearSVC(C=500)
svm_classifier.fit(train_data, train_result)
test_error = util.measureTestError(test_data, test_result, svm_classifier)
print 'Test error with c=500', test_error
print '------------------------------PART 2 Cross Validation Error,\
 Training error, test Error with different C s ------------------------------------------'
cs = [10,10**2,10**3,10**4,5*(10**5),10**6]
logcs = [math.log(c) for c in cs]
cross_validation_errors = []
train_errors = []
test_errors = []
for c in cs:
    print 'c:', c
    cross_validation_error, train_error = doNFoldCrossValidation(10, train_data, train_result, c)
    svm_classifier = svm.LinearSVC(C=c)
    svm_classifier.fit(train_data, train_result)
    test_error = util.measureTestError(test_data, test_result, svm_classifier)
    cross_validation_errors.append(cross_validation_error)
    train_errors.append(train_error)
    test_errors.append(test_error)
util.plotErrorForK(logcs, cross_validation_errors, train_errors, test_errors, algo='SVM')
print '--------------------------------------------------------------'