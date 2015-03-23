__author__ = 'santhosh'
import numpy
import math


class DecisionStump(object):
    def __init__(self):
        self.root = (-1, None, None, []) #feature_idx, feature_value_threshold, result_instances, child_nodes

    def determineFeatureIdxAndThreshold(self, train_data, train_results, weights):
        return -1, -1

    def fit(self, train_data, train_results, weights):
        n_samples, n_features = train_data.shape
        feature_idx, feature_value_threshold =\
            self.determineFeatureIdxAndThreshold(train_data,\
                                                 train_results,\
                                                 weights)

        total_negative_values, total_positive_values = 0

        for i in range(n_samples):
            if train_results[i] == 1:
                total_negative_values += 1
            else:
                total_positive_values += 1

        self.root = (feature_idx, feature_value_threshold,(total_negative_values, total_positive_values), [])
        feature_idx, feature_value_threshold, result_instances, child_nodes = self.root
        total_negative_values_for_less, total_positive_values_for_less = 0, 0
        total_negative_values_for_greater, total_positive_values_for_greater = 0, 0

        for i in range(n_samples):
            if train_data[feature_idx] < feature_value_threshold:
                if train_results[i] == 1:
                    total_negative_values_for_less += 1
                else:
                    total_positive_values_for_less += 1
            else:
                if train_results[i] == 1:
                    total_negative_values_for_greater += 1
                else:
                    total_positive_values_for_greater += 1

        child_node0 = (-1, None, (total_negative_values_for_less, total_positive_values_for_less), None)
        child_node1 = (-1, None, (total_negative_values_for_greater, total_positive_values_for_greater), None)

        child_nodes.append(child_node0)
        child_nodes.append(child_node1)
        self.root = (feature_idx, feature_value_threshold,\
                     (total_negative_values, total_positive_values),\
                     child_nodes)

    def predict(self, test_data_instance):
        feature_idx, feature_value_threshold, result_instances, child_nodes = self.root
        if test_data_instance[feature_idx] < feature_value_threshold:
            result_instance_counts = child_nodes[0][3]
            return -1 if result_instance_counts[0] > result_instance_counts[1] else 1
        else:
            result_instance_counts = child_nodes[1][3]
            return -1 if result_instance_counts[0] > result_instance_counts[1] else 1


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
            final_value += self.alphas[i]*self.weakClassifiers[i].predict(test_data_instance)
        return 1 if final_value < 0 else 2