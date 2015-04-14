__author__ = 'santhosh'

import matplotlib.pyplot as plt
import os


def measureTestError(test_data, test_result, classifier):
    test_samples, test_features = test_data.shape
    test_error = 0.0
    for i in range(test_samples):
        label = classifier.predict(test_data[i])
        if label != test_result[i]:
            test_error += 1.0
    test_error /= test_samples
    return test_error



def plotErrorForK(ks, errors, train_errors, test_errors, algo = 'KNN'):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.title('Error Plot')
    plt.xlabel('Parameter - LOG(C) or K')
    plt.ylabel('Error')
    ax.plot(ks, errors, label='CrossValidation Error', color='r')
    ax.plot(ks, train_errors, label='Train Error', color='b')
    ax.plot(ks, test_errors, label='Test Error', color='g')
    imgFile = os.path.join(os.getcwd(), algo+"Error plot")+'.png'
    art = []
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    art.append(lgd)
    plt.tight_layout()
    print algo+" Error plot logged to "+imgFile
    plt.savefig(imgFile,\
                 bbox_inches="tight")
    plt.close()
