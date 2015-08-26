__author__ = 'santhosh'
import numpy

class NeuralNets(object):
    def __init__(self, nodes_in_hidden_layers, hidden_layer_fn_and_diff_matrix, step_size, threshold):
        self.nodes_in_hidden_layers = nodes_in_hidden_layers
        self.hidden_layer_fn_and_diff = numpy.array(hidden_layer_fn_and_diff_matrix)
        self.step_size = step_size
        self.threshold = threshold

    def train(self, train_data):
        no_of_train_samples, no_of_features = train_data.shape

    def predict(self, test_data):
        pass