__author__ = 'santhosh'

import numpy

class KNN(object):
    def __init__(self):
        self.train_data = []
        self.train_sample_pair_wise_distance_matrix = {}
        self.train_result = []
        self. k = 10
        
    def cosine_distance(self,x,y):
        denom = numpy.sqrt(numpy.dot(x,x)*numpy.dot(y,y))
        distance = 1 -(numpy.dot(x,y)/denom)
        return distance
    
    def fixK(self, k):
        self.k = k
    
    def calculatePairWiseDistanceForTrainSamples(self):
        n_samples, n_features = self.train_data.shape
        self.train_sample_pair_wise_distance_matrix = {i:{j:self.cosine_distance(self.train_data[i], self.train_data[j])\
                                                           for j in range(n_samples) if i != j} for i in range(n_samples)}
        
    def findKNearestNeigbors(self, x, k = None, leaveIndicesOnTrainSamples = []):
        if k == None:
            k = self.k
        n_samples, n_features = self.train_data.shape
        neighbor_distances_dict = dict()
        for i in range(n_samples):
            if i not in leaveIndicesOnTrainSamples:
                train_vector = self.train_data[i]
                distance_for_this_sample = self.cosine_distance(train_vector, x)
                neighbor_distances_dict[i] = distance_for_this_sample
        
        neighbors_array = sorted(list(neighbor_distances_dict.keys()), key = lambda key: neighbor_distances_dict[key])
        return neighbors_array[:k]
    
    def fit(self, X, y):
        self.train_data = X
        self.train_result = y
        self.calculatePairWiseDistanceForTrainSamples()
        for i in range(100):
            self.k = i
            
    
    def predict(self, x):
        knn  = self.findKNearestNeigbors(x)
        class_labels = dict()
        for i in range(len(knn)):
            sample_idx = knn[i]
            label = self.train_result[sample_idx]
            if label not in class_labels:
                class_labels[label] = 0
            class_labels[label] += 1
            
        return max(class_labels.keys(), key = lambda key: class_labels[key])

