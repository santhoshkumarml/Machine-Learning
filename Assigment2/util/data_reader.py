__author__ = 'santhosh'

import csv
import numpy

def read_data():
    with open('../bupa.data','r') as f:
        csv_data = csv.reader(f, delimiter=',')
        data  = []
        result = []
        for row in csv_data:
            result.append(row[-1])
            data.append(row[:len(row)-1])
        data = numpy.array(data, dtype=float)
        result = numpy.array(result)
        n_samples,n_features = data.shape
        train_sample_size = int(n_samples*0.9)
        train_data,train_result  = data[:train_sample_size],result[:train_sample_size]
        test_data,test_result = data[train_sample_size:],result[train_sample_size:]
        return train_data, train_result, test_data, test_result

