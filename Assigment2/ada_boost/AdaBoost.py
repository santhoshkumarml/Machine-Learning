__author__ = 'santhosh'
import numpy


class DecisionStump(object):
    def __init__(self):
        pass

    def fit(self, train_data, train_results, weights):
        pass

    def predict(self, test_data_instance):
        pass


class AdaBoostClassifier(object):
    def __init__(self,iterations=100):
        self.iterations = iterations
        self.weakClassifiers = [DecisionStump() for i in range(self.iterations)]
        self.alphas = numpy.zeros(self.iterations)

    def fit(self, train_data, train_result):
        n_samples, n_features = train_data.shape
        weights = numpy.array([(1.0/n_samples) for i in range(n_samples)])
        for i in range(self.iterations):
            self.weakClassifiers[i].fit(train_data, train_result, weights)

    def predict(self, test_data_instance):
        final_value = 0
        for i in range(self.iterations):
            final_value += self.alphas[i]* self.weakClassifiers[i].predict(test_data_instance)
        return -1 if final_value < 0 else 1