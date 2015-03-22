__author__ = 'santhosh'
import numpy
import math


class DecisionStump(object):
    def __init__(self):
        pass

    def fit(self, train_data, train_results, weights):
        pass

    def predict(self, test_data_instance):
        pass


class AdaBoostClassifier(object):
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.weakClassifiers = [DecisionStump() for i in range(self.iterations)]
        self.alphas = numpy.zeros(self.iterations)

    def fit(self, train_data, train_result):
        n_samples, n_features = train_data.shape
        weights = numpy.array([(1.0/n_samples) for i in range(n_samples)])
        for i in range(self.iterations):
            self.weakClassifiers[i].fit(train_data, train_result, weights)
            # determine alpha_t
            training_error_results = [1 if train_result[i] !=\
                                          self.weakClassifiers[i].predict(train_data[i]) else 0]
            training_error = float(sum(training_error_results))/n_samples
            inner_calc_for_alpha = (1-training_error)/training_error
            self.alphas[i] = 0.5*math.log(inner_calc_for_alpha, math.exp(1))
            # redetermine weights for next round
            new_weights = [weights[i]*\
                           math.exp(-1*self.alphas[i]*train_result[i]*\
                                    self.weakClassifiers[i].predict(train_data[i]))\
                           for i in range(n_samples)]
            z_t = sum(new_weights)
            weights = numpy.array([new_weights[i]/z_t for i in range(n_samples)])

    def predict(self, test_data_instance):
        final_value = 0
        for i in range(self.iterations):
            final_value += self.alphas[i]* self.weakClassifiers[i].predict(test_data_instance)
        return -1 if final_value < 0 else 1