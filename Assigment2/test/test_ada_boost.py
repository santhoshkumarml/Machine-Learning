__author__ = 'santhosh'

from util import data_reader
from ada_boost.AdaBoost import AdaBoostClassifier

train_data, train_result, test_data, test_result = data_reader.read_data()
adc = AdaBoostClassifier(train_data, train_result)
adc.predict(test_data)