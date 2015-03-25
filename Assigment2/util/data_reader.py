__author__ = 'santhosh'

import csv
import numpy
import random


def read_data(data_split = 0.9):
    with open('../bupa.data','r') as f:
        csv_data = csv.reader(f, delimiter=',')
        data = []
        result = []
        for row in csv_data:
            result.append(row[-1])
            data.append(row[:len(row)-1])
        data = numpy.array(data, dtype=float)
        result = numpy.array(result)
        n_samples, n_features = data.shape
        train_sample_size = int(n_samples*data_split)
        # random_indexes_for_train = range(train_sample_size)
        # random_indexes_for_test = list(set(range(n_samples))-set(random_indexes_for_train))
        random_indexes_for_train = random.sample(xrange(n_samples), train_sample_size)
        random_indexes_for_test = list(set(range(n_samples))-set(random_indexes_for_train))

        train_data, train_result, test_data, test_result = [], [], [], []

        for i in random_indexes_for_train:
            train_data.append(data[i])
            train_result.append(result[i])
        for i in random_indexes_for_test:
            test_data.append(data[i])
            test_result.append(result[i])

        return numpy.array(train_data), numpy.array(train_result), numpy.array(test_data), numpy.array(test_result)

