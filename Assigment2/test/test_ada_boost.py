__author__ = 'santhosh'

from util import data_reader
from ada_boost.AdaBoost import AdaBoostClassifier

train_data, train_result, test_data, test_result = data_reader.read_data()
# import numpy
# train_data = numpy.array([[2,0],[2,2],[1,1],[3,1]])
# train_data = numpy.array([[0,-1],[0,1],[1,0],[-1,0]])
# train_result = numpy.array(['1','1','2','2'])
# test_data = train_data
# test_result = train_result
adc = AdaBoostClassifier()

train_error_for_this_sample, test_error_for_this_sample = \
    adc.fitPredictAndScore(train_data, train_result, test_data, test_result)

print train_error_for_this_sample, test_error_for_this_sample