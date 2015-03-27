__author__ = 'santhosh'

from util import data_reader
from ada_boost.AdaBoost import AdaBoostClassifier
import os
import datetime

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

def question1(no_of_random_samples, T):
    print '--------------------------------------------------'
    print 'Question 1: going to run '+str(T)+' boosting iterations on '\
          + str(no_of_random_samples)+' random samples using 90% of data as train data and 10% as test data'
    before = datetime.datetime.now()
    adc = AdaBoostClassifier(T)
    avg_train_error, avg_test_error = [0 for i in range(T)], [0 for i in range(T)]
    for i in range(no_of_random_samples):
        train_data, train_result, test_data, test_result = data_reader.read_data(data_split=0.9)
        print 'Random Sample ', str(i + 1)
        train_error_for_iters, test_error_for_iters = adc.fitPredictAndScore(train_data, train_result, test_data,
                                                                             test_result, calc_error_each_iter=True)
        print 'Train Error:', str(train_error_for_iters[-1]), ' Test Error:', str(test_error_for_iters[-1])
        for j in range(len(train_error_for_iters)):
            avg_train_error[j] += train_error_for_iters[j]
            avg_test_error[j] += test_error_for_iters[j]
    avg_train_error = [avg_train_error[i] / no_of_random_samples for i in range(T)]
    avg_test_error = [avg_test_error[i] / no_of_random_samples for i in range(T)]
    plotError(avg_train_error, 'Avg Train Error',
              'Avg Train Error For ' + str(no_of_random_samples) + ' random splits and ' + str(T) + ' iterations')
    plotError(avg_test_error, 'Avg Test Error',
              'Avg Test Error For ' + str(no_of_random_samples) + ' random splits and ' + str(T) + ' iterations')
    print 'Question 1 Done - Average Train and Test Error For ' + str(
        no_of_random_samples) + ' random splits and ' + str(T) + ' iterations of boosting'
    after = datetime.datetime.now()
    print 'Time Taken:', after - before
    print '--------------------------------------------------'


def question2(T):
    # question 2
    print '--------------------------------------------------'
    print 'Question 2: going to run '+str(T)+' iterations '\
          +'using all of the data as training data'
    before = datetime.datetime.now()
    adc = AdaBoostClassifier(T)
    train_data, train_result, test_data, test_result = data_reader.read_data(data_split=1)
    adc.fitPredictAndScore(train_data, train_result, test_data, test_result, showDecisionStump=range(10))
    after = datetime.datetime.now()
    print 'Question 2 Done'
    print 'Time Taken:', after - before
    print '--------------------------------------------------'


question1(50,100)
question2(100)