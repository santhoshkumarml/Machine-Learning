__author__ = 'santhosh'
import numpy

class DecisionStump(object):
    def __init__(self, data):
        pass

class AdaBoostClassifier(object):
    def __init__(self, train_data, train_result):
        n_samples,n_features = train_data.shape
        weights = numpy.array([(1.0/n_samples) for i in range(n_samples)])
        self.weighted_data = numpy.column_stack((train_data,weights))
        self.results = train_result

    def predict(self, test_data):
        print self.weighted_data.shape, self.results.shape