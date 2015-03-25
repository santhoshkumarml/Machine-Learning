__author__ = 'santhosh'

from util import data_reader
from ada_boost.AdaBoost import AdaBoostClassifier
import os

def plotError(error_per_iteration, pltFile, label):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title(label)
    plt.xlabel('Iteration')
    plt.ylabel('Error ')
    ax.plot(error_per_iteration,'g')
    imgFile = os.path.join(os.pardir, pltFile)+'.png'
    print label+" plot logged to "+os.path.abspath(imgFile)
    plt.savefig(imgFile)
    plt.close()

# import numpy
# train_data = numpy.array([[0,-1],[0,1],[1,0],[-1,0]])
# train_result = numpy.array(['1','1','2','2'])
# test_data = train_data
# test_result = train_result
# adc = AdaBoostClassifier(4)
# adc.fitPredictAndScore(train_data, train_result, test_data, test_result)
#
# import sys
# sys.exit()

# question 1
print '--------------------------------------------------'
# T = 100
# adc = AdaBoostClassifier(T)
# avg_train_error, avg_test_error = [0 for i in range(T)], [0 for i in range(T)]
# no_of_random_samples = 5
# for i in range(no_of_random_samples):
#     train_data, train_result, test_data, test_result = data_reader.read_data(data_split=0.9)
#     train_error_for_iters, test_error_for_iters = adc.fitPredictAndScore(train_data, train_result, test_data, test_result)
#     print 'Sample ',str(i+1),' Train Error:', str(train_error_for_iters[-1]),' Test Error:',str(test_error_for_iters[-1])
#     for j in range(len(train_error_for_iters)):
#         avg_train_error[j] += train_error_for_iters[j]
#         avg_test_error[j] += test_error_for_iters[j]
#
# avg_train_error = [avg_train_error[i]/no_of_random_samples for i in range(T)]
# avg_test_error = [avg_test_error[i]/no_of_random_samples for i in range(T)]
#
# plotError(avg_train_error, 'Avg Train Error', 'Avg Train Error For '+str(no_of_random_samples)+' random splits and '+str(T)+' iterations of boosting')
# plotError(avg_test_error, 'Avg Test Error', 'Avg Test Error For '+str(no_of_random_samples)+' random splits and '+str(T)+' iterations of boosting')
# print 'Question 1 Done - Average Train and Test Error For '+str(no_of_random_samples)+' random splits and '+str(T)+' iterations of boosting'
print '--------------------------------------------------'

# question 2
print '--------------------------------------------------'
T = 100
adc = AdaBoostClassifier(T)
train_data, train_result, test_data, test_result = data_reader.read_data(data_split=1)
train_error_for_iters, test_error_for_iters =\
    adc.fitPredictAndScore(train_data, train_result, test_data, test_result, showDecisionStump=range(10))
print 'Question 2 Done'
print '--------------------------------------------------'