__author__ = 'santhosh'

from util import data_reader
from ada_boost.AdaBoost import AdaBoostClassifier

train_data, train_result, test_data, test_result = data_reader.read_data()
adc = AdaBoostClassifier()
adc.fit(train_data, train_result)
for i in range(len(test_data)):
    result = adc.predict(test_data[i])
    print result, test_result[i]
