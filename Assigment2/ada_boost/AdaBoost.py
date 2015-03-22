__author__ = 'santhosh'
import numpy
import math


class DecisionNode(object):
    def __init__(self):
        self.feature_idx = -1
        self.feature_threshold = None
        self.result_probability = dict()
        self.childNodes = []

    def fit(self, feature_idx, feature_threshold, result_probability):
        self.feature_idx = feature_idx
        self.feature_threshold = feature_threshold

    def getChildNodes(self):
        return self.childNodes


class DecisionStump(object):
    def __init__(self):
        self.root = (-1, None, dict(), []) #feature_idx, feature_value_threshold, result_instances, child_nodes

    def determineFeatureIdxAndThreshold(self):
        return -1, -1

    def fit(self, train_data, train_results, weights):
        feature_idx, feature_value_threshold = self.determineFeatureIdxAndThreshold()
        feature_idx, feature_value_threshold, result_instances, child_nodes = self.root
        child_nodes.append((-1, None, dict(), []))
        child_nodes.append((-1, None, dict(), []))

    def predict(self, test_data_instance):
        feature_idx, feature_value_threshold, result_instances, child_nodes = self.root
        if test_data_instance[feature_idx] < feature_value_threshold:
            result_instance_counts = child_nodes[0][3]
            return -1 if result_instance_counts[0]>result_instance_counts[1] else 1
        else:
            result_instance_counts = child_nodes[1][3]
            return -1 if result_instance_counts[0]>result_instance_counts[1] else 1


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