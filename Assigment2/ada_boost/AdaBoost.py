__author__ = 'santhosh'
import numpy
import math
import random


class DecisionStump(object):
    def __init__(self):
        self.root = (-1, None, None, []) #feature_idx, feature_value_threshold, result_instances, child_nodes

    def getTrainResultClass(self, train_result_value):
        return -1 if train_result_value == '1' else 1

    def buildStump(self, feature_idx, feature_value_threshold, train_data, train_results):
        n_samples, n_features = train_data.shape
        total_negative_values, total_positive_values = 0, 0
        for i in range(n_samples):
            if train_results[i] == '1':
                total_negative_values += 1
            else:
                total_positive_values += 1
        root = (feature_idx, feature_value_threshold, (total_negative_values, total_positive_values), [])
        feature_idx, feature_value_threshold, result_instances, child_nodes = root
        total_negative_values_for_less, total_positive_values_for_less = 0, 0
        total_negative_values_for_greater, total_positive_values_for_greater = 0, 0
        for i in range(n_samples):
            if train_data[i][feature_idx] < feature_value_threshold:
                if train_results[i] == '1':
                    total_negative_values_for_less += 1
                else:
                    total_positive_values_for_less += 1
            else:
                if train_results[i] == '1':
                    total_negative_values_for_greater += 1
                else:
                    total_positive_values_for_greater += 1
        child_node0 = (-1, None, (total_negative_values_for_less, total_positive_values_for_less), None)
        child_node1 = (-1, None, (total_negative_values_for_greater, total_positive_values_for_greater), None)
        child_nodes.append(child_node0)
        child_nodes.append(child_node1)
        root = (feature_idx, feature_value_threshold, \
                     (total_negative_values, total_positive_values), child_nodes)
        return root

    def determineFeatureIdxAndThreshold(self, train_data, train_results, weights):
        n_samples, n_features = train_data.shape
        weighted_classification_error = [(float('inf'), -1) for i in range(n_features)]
        step = 1
        for i in range(n_features):
            train_data_for_feature = train_data[:, i]
            min_val = min(train_data_for_feature)
            max_val = max(train_data_for_feature)
            for step_val in numpy.arange(min_val, max_val+step, step):
                error = 0
                root = self.buildStump(i, step_val, train_data, train_results)
                for j in range(n_samples):
                    prediction = self.predict_with_root(root, train_data[j])
                    if prediction != self.getTrainResultClass(train_results[j]):
                        error += weights[j]
                if error < weighted_classification_error[i][0]:
                    weighted_classification_error[i] = (error, step_val)

        print weighted_classification_error
        optimal_feature_idx = min(range(n_features), key = lambda idx: weighted_classification_error[idx][0])
        optimal_feature_threshold = weighted_classification_error[optimal_feature_idx][1]
        return optimal_feature_idx, optimal_feature_threshold

    def fit(self, train_data, train_results, weights):
        n_samples, n_features = train_data.shape
        feature_idx, feature_value_threshold =\
            self.determineFeatureIdxAndThreshold(train_data,\
                                                 train_results,\
                                                 weights)
        self.root = self.buildStump(feature_idx, feature_value_threshold, train_data, train_results)

    def predict_with_root(self, root, test_data_instance):
        feature_idx, feature_value_threshold, result_instances, child_nodes = root
        if test_data_instance[feature_idx] < feature_value_threshold:
            result_instance_counts = child_nodes[0][2]
            return -1 if result_instance_counts[0] > result_instance_counts[1] else 1
        else:
            result_instance_counts = child_nodes[1][2]
            return -1 if result_instance_counts[0] > result_instance_counts[1] else 1

    def predict(self, test_data_instance):
        return self.predict_with_root(self.root, test_data_instance)

    def getRoot(self):
        return self.root


class AdaBoostClassifier(object):
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.weakClassifiers = [DecisionStump() for i in range(self.iterations)]
        self.alphas = numpy.zeros(self.iterations)
        self.z_t_s = numpy.zeros(self.iterations)

    def getTrainResultClass(self, train_result_value):
        return -1 if train_result_value == '1' else 1

    def predict(self, test_data_instance, iter = -1):
        final_value = 0
        if iter == -1:
            iter = self.iterations
        for t in range(iter):
            final_value += self.alphas[t]*self.weakClassifiers[t].predict(test_data_instance)
        return '1' if final_value < 0 else '2'

    def fitPredictAndScore(self, train_data, train_result, test_data, test_result):

        n_samples, n_features = train_data.shape
        weights = numpy.array([(1.0/n_samples) for i in range(n_samples)])
        training_error_in_iterations, test_error_in_iterations = [],[]

        for t in range(self.iterations):
            self.weakClassifiers[t].fit(train_data, train_result, weights)
            print 'Decision Stump at iter', t, ':', self.weakClassifiers[t].getRoot()
            # determine alpha_t
            training_error_results = [weights[i] if self.getTrainResultClass(train_result[i]) !=\
                                           self.weakClassifiers[t].predict(train_data[i])\
                                          else 0 for i in range(n_samples)]
            training_error = sum(training_error_results)
            print 'Weighted Training Error on iter ', t, ':', training_error
            training_error_in_iterations.append(training_error)
            inner_calc_for_alpha = (1-training_error)/training_error
            self.alphas[t] = 0.5*math.log(inner_calc_for_alpha, math.exp(1))
            # redetermine weights for next round
            new_weights = [weights[i]*\
                           math.exp(-1*self.alphas[t]*self.getTrainResultClass(train_result[i])*\
                                    self.weakClassifiers[t].predict(train_data[i]))\
                           for i in range(n_samples)]
            z_t = sum(new_weights)
            weights = numpy.array([new_weights[i]/z_t for i in range(n_samples)])
            self.z_t_s[t] = z_t
            error = 0
            for tidx in range(len(test_data)):
                test_data_ins = test_data[tidx]
                test_result_ins = test_result[tidx]
                prediction = self.predict(test_data_ins, iter = t+1)
                if prediction != test_result_ins:
                    error += 1
            test_error = float(error)/len(test_data)
            print 'Test Error on iter ', t, ':', test_error
            test_error_in_iterations.append(test_error)
        return training_error_in_iterations, test_error_in_iterations

