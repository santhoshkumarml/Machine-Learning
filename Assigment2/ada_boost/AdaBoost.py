__author__ = 'santhosh'
import numpy
import math


class DecisionStump(object):
    def __init__(self):
        self.root = (-1, None, None, []) #feature_idx, feature_value_threshold, result_instances, child_nodes

    def getTrainResultClass(self, train_result_value):
        return -1 if train_result_value == '1' else 1

    def buildStump(self, feature_idx, feature_value_threshold, train_data, train_results):
        n_samples, n_features = train_data.shape
        total_negative_values, total_positive_values = 0, 0
        root1 = (feature_idx, feature_value_threshold, (-1, 1))
        root2 = (feature_idx, feature_value_threshold, (1, -1))
        error_for_root1, error_for_root2 = 0, 0
        for i in range(n_samples):
            if self.getTrainResultClass(train_results[i]) != self.predict_with_root(root1, train_data[i]):
                error_for_root1 += 1
            if self.getTrainResultClass(train_results[i]) != self.predict_with_root(root2, train_data[i]):
                error_for_root2 += 1
        return root1 if error_for_root1<error_for_root2 else root2

    def determineBestStump(self, train_data, train_results, weights):
        n_samples, n_features = train_data.shape
        min_weighted_classification_error = (1, None)
        for feature_idx in range(n_features):
            min_val = min(train_data[:, feature_idx])
            max_val = max(train_data[:, feature_idx])
            step = 1
            for step_val in numpy.arange(min_val, max_val, step):
                error = 0
                root = self.buildStump(feature_idx, step_val, train_data, train_results)
                for j in range(n_samples):
                    prediction = self.predict_with_root(root, train_data[j])
                    if prediction != self.getTrainResultClass(train_results[j]):
                        error += weights[j]
                if error < min_weighted_classification_error[0]:
                    min_weighted_classification_error= (error, root)
            #print 'all weighted errors for feature', feature_idx, errors_for_each_step_val, min(errors_for_each_step_val, key=lambda e: e[0])

        #print weighted_classification_error
        error, root = min_weighted_classification_error
        return root

    def fit(self, train_data, train_results, weights):
        n_samples, n_features = train_data.shape
        self.root = self.determineBestStump(train_data, train_results, weights)

    def predict_with_root(self, root, test_data_instance):
        feature_idx, feature_value_threshold, child_leaves = root
        return child_leaves[0] if test_data_instance[feature_idx] <= feature_value_threshold else child_leaves[1]

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

    def predict(self, test_data_instance, iter=-1):
        final_value = 0
        if iter == -1:
            iter = self.iterations
        for t in range(iter):
            final_value += self.alphas[t]*self.weakClassifiers[t].predict(test_data_instance)
        return '1' if final_value < 0 else '2'

    def fitPredictAndScore(self, train_data, train_result, test_data, test_result):
        n_samples, n_features = train_data.shape
        weights = numpy.array([(1.0/n_samples) for i in range(n_samples)])
        training_error_in_iterations, test_error_in_iterations = [], []

        for t in range(self.iterations):
            #print 'Weights of samples in this iteration', weights
            self.weakClassifiers[t].fit(train_data, train_result, weights)
            #print 'Decision Stump at iter', t, ':', self.weakClassifiers[t].getRoot()
            # determine alpha_t
            weighted_training_error = 0
            correctly_classified_instances = set()
            for i in range(n_samples):
                if self.weakClassifiers[t].predict(train_data[i]) != self.getTrainResultClass(train_result[i]):
                    weighted_training_error += weights[i]
                else:
                    correctly_classified_instances.add(i)

            #print 'Weighted Training Error on iter ', t, ':', weighted_training_error
            inner_calc_for_alpha = (1-weighted_training_error)/weighted_training_error
            self.alphas[t] = 0.5*math.log(inner_calc_for_alpha)
            # redetermine weights for next round
            new_weights = numpy.zeros(n_samples)
            for i in range(n_samples):
                # print 'NotClassified' if i not in correctly_classified_instances else 'Correctly Classified'
                # print self.getTrainResultClass(train_result[i]), self.weakClassifiers[t].predict(train_data[i])
                # print -1*self.alphas[t]*self.getTrainResultClass(train_result[i])* self.weakClassifiers[t].predict(train_data[i])
                new_weights[i] = weights[i]*math.exp(-1*self.alphas[t]*\
                                                     self.getTrainResultClass(train_result[i])*\
                                                     self.weakClassifiers[t].predict(train_data[i]))
                # print i, weights[i], new_weights[i]

            z_t = sum(new_weights)
            #
            # for i in range(n_samples):
            #     if i not in correctly_classified_instances:
            #         print 'Normalised NotClassified'
            #     else:
            #         print 'Normalized Correctly Classified'
            #     print i, weights[i], new_weights[i]/z_t

            weights = numpy.array([new_weights[i]/z_t for i in range(n_samples)])
            self.z_t_s[t] = z_t
            training_error = 0
            for tidx in range(len(train_data)):
                train_data_ins = train_data[tidx]
                train_result_ins = train_result[tidx]
                prediction = self.predict(train_data_ins, iter=t+1)
                if prediction != train_result_ins:
                    training_error += 1
            training_error = float(training_error)/len(train_data)
            training_error_in_iterations.append(training_error)
            #print 'Overall Train Error on iter ', t, ':', training_error
            error = 0
            for tidx in range(len(test_data)):
                test_data_ins = test_data[tidx]
                test_result_ins = test_result[tidx]
                prediction = self.predict(test_data_ins, iter=t+1)
                if prediction != test_result_ins:
                    error += 1
            test_error = float(error)/len(test_data)
            #print 'Overall Test Error on iter ', t, ':', test_error
            test_error_in_iterations.append(test_error)
        return training_error_in_iterations, test_error_in_iterations