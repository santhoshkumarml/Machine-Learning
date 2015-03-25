__author__ = 'santhosh'

from util import data_reader
from ada_boost.AdaBoost import AdaBoostClassifier

def plotError(error_per_iteration, label):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    plt.title(label+' error Plot')
    plt.xlabel('Iteration')
    plt.ylabel('Error ')
    ax.plot(error_per_iteration,'g')
    plt.show()

train_data, train_result, test_data, test_result = data_reader.read_data()
# import numpy
# train_data = numpy.array([[2,0],[2,2],[1,1],[3,1]])
# train_data = numpy.array([[0,-1],[0,1],[1,0],[-1,0]])
# train_result = numpy.array(['1','1','2','2'])
# test_data = train_data
# test_result = train_result
T = 100
adc = AdaBoostClassifier(T)
avg_train_error, avg_test_error = [0 for i in range(T)], [0 for i in range(T)]
no_of_random_samples = 3
for i in range(no_of_random_samples):
    train_error_for_iters, test_error_for_iters = adc.fitPredictAndScore(train_data, train_result, test_data, test_result)
    for j in range(len(train_error_for_iters)):
        avg_train_error[j] += train_error_for_iters[j]
        avg_test_error[j] += test_error_for_iters[j]

avg_train_error = [avg_train_error[i]/no_of_random_samples for i in range(len(avg_train_error))]
avg_test_error = [avg_test_error[i]/no_of_random_samples for i in range(len(avg_test_error))]

plotError(avg_train_error, 'Train')
plotError(avg_test_error, 'Test')
