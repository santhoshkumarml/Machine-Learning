__author__ = 'santhosh'

import numpy
import matplotlib.pyplot as plt
import os
import random


def plotErrorForK(ks, errors, train_errors, test_errors):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title('Error Plot')
    plt.xlabel('K')
    plt.ylabel('Error')
    ax.plot(ks, errors, label='CrossValidation Error', color='r')
    ax.plot(ks, train_errors, label='Train Error', color='b')
    ax.plot(ks, test_errors, label='Test Error', color='g')
    imgFile = os.path.join(os.getcwd(), "KNN Error plot")+'.png'
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    art.append(lgd)
    plt.tight_layout()
    print "Error plot logged to "+imgFile
    plt.savefig(imgFile,\
                 bbox_inches="tight")
    plt.close()

class KNNClassifier(object):
    def __init__(self, possible_k_values, cross_validation_fold):
        self.train_data = []
        self.train_sample_pair_wise_distance_matrix = {}
        self.train_result = []
        self.nFold = cross_validation_fold
        self.possible_k_values = possible_k_values

    def cosine_distance(self,x,y):
        denom = numpy.sqrt(numpy.dot(x,x)*numpy.dot(y,y))
        distance = 1 -(numpy.dot(x,y)/denom)
        return distance
    
    def calculatePairWiseDistanceForTrainSamples(self):
        n_samples, n_features = self.train_data.shape
        for i in range(n_samples):
            neighbor_distances = {j:self.cosine_distance(self.train_data[i], self.train_data[j])\
                                  for j in range(n_samples) if i != j}
            neighbor_distances = sorted(list(neighbor_distances.keys()),\
                key = lambda key : neighbor_distances[key])
            self.train_sample_pair_wise_distance_matrix[i] = neighbor_distances
        
    def findKNearestNeigbors(self, x, k):
        n_samples, n_features = self.train_data.shape
        neighbor_distances_dict = dict()
        for i in range(n_samples):
            train_vector = self.train_data[i]
            distance_for_this_sample = self.cosine_distance(train_vector, x)
            neighbor_distances_dict[i] = distance_for_this_sample
        neighbors_array = sorted(list(neighbor_distances_dict.keys()),\
            key = lambda key: neighbor_distances_dict[key])
        return neighbors_array[:k]

    def doCrossValiationForK(self, k, test_idx_for_each_iter):
        n_samples, n_features = self.train_data.shape
        cross_validation_error = 0.0
        train_error = 0.0
        for sample_idx in range(n_samples):
            knn_for_train = [neighbor for neighbor in self.train_sample_pair_wise_distance_matrix[sample_idx]][:k]
            label = self.getMajorityClassLabelsForKNN(knn_for_train, \
                                                      self.train_data[sample_idx], \
                                                      self.train_result[sample_idx])
            if label != self.train_result[sample_idx]:
                train_error += 1.0
        for i in range(len(test_idx_for_each_iter)):
            test_idxs = test_idx_for_each_iter[i]
            train_data_for_iter, train_result_for_iter = [], []
            for sample_idx in range(n_samples):
                if sample_idx not in test_idxs:
                    train_data_for_iter.append(self.train_data[sample_idx])
                    train_result_for_iter.append(self.train_result[sample_idx])
            for test_idx in test_idxs:
                knn = [neighbor for neighbor in \
                       self.train_sample_pair_wise_distance_matrix[test_idx] \
                       if neighbor not in test_idxs][:k]
                label = self.getMajorityClassLabelsForKNN(knn, train_data_for_iter, train_result_for_iter)
                test_label = self.train_result[test_idx]
                if label != test_label:
                    cross_validation_error += 1
        return cross_validation_error, train_error

    def doNFoldCrossValidation(self):
        n_samples, n_features = self.train_data.shape
        partition_size = n_samples/self.nFold
        test_idx_for_each_iter = [set() for fold in range(self.nFold)]
        random_idxs = range(n_samples)
        # random_idxs = random.sample(xrange(n_samples), n_samples)
        print random_idxs
        itr = 0
        for i in range(0, n_samples):
            if len(test_idx_for_each_iter[itr]) != partition_size:
                test_idx_for_each_iter[itr].add(random_idxs[i])
            else:
                itr+=1
        cross_validation_error_for_each_k = dict()
        train_error_for_each_k = []

        for k in self.possible_k_values:
            cross_validation_error, train_error = self.doCrossValiationForK(k, test_idx_for_each_iter)
            cross_validation_error_for_each_k[k] = cross_validation_error/n_samples
            train_error /= n_samples
            train_error_for_each_k.append(train_error)

        self.k = min(cross_validation_error_for_each_k.keys(),\
            key = lambda key: cross_validation_error_for_each_k[key])
        print 'Minimum error with k: ', self.k

        return cross_validation_error_for_each_k.values(), train_error_for_each_k

    def getMajorityClassLabelsForKNN(self, knn, train_data, train_result):
        class_labels = dict()
        for i in range(len(knn)):
            sample_idx = knn[i]
            label = self.train_result[sample_idx]
            if label not in class_labels:
                class_labels[label] = 0
            class_labels[label] += 1
        return max(class_labels.keys(), key=lambda key: class_labels[key])

    def predict(self, x, k):
        knn = self.findKNearestNeigbors(x, k)
        return self.getMajorityClassLabelsForKNN(knn, self.train_data, self.train_result)

    def score(self, labels, result):
        error = 0.0
        for i in range(len(labels)):
            if labels[i] != result[i]:
                error += 1.0
        error /= len(labels)
        return error

    def fitPredictAndScore(self, train_data, train_result, test_data, test_result):
        self.train_data = train_data
        n_samples, n_features = self.train_data.shape

        self.train_result = train_result
        self.calculatePairWiseDistanceForTrainSamples()

        cross_validation_error_for_each_k, train_error_for_each_k = self.doNFoldCrossValidation()
        test_error_for_each_k = []
        knn_for_test_data = {}

        for test_idx in range(len(test_data)):
            knn_for_test_data[test_idx] = self.findKNearestNeigbors(test_data[test_idx], k=n_samples)

        for k in self.possible_k_values:
            labels = [self.getMajorityClassLabelsForKNN(knn_for_test_data[test_idx][:k],\
                                                        self.train_data, self.train_result)\
                      for test_idx in range(len(test_data))]
            test_error = self.score(labels, test_result)
            test_error_for_each_k.append(test_error)
        plotErrorForK(self.possible_k_values, cross_validation_error_for_each_k,\
                      train_error_for_each_k, test_error_for_each_k)
