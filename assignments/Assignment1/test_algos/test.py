'''
Created on Feb 20, 2015

@author: santhosh
'''
from util import datareader
from naivebayes.NaiveBayes import NaiveBayesClassifier
from logisticregression.LogisticRegression import LogisticRegression
from datetime import datetime

def fitAndPredict(clf):
    X,y = datareader.readTrainDataAndLabels()
    clf.fit(X, y)
    accuracy = 0.0

    for i in range(X.shape[0]):
        label = clf.predict(X[i])
        if label == y[i]:
            accuracy += 1.0
    accuracy1 = accuracy/len(y)

    X = datareader.readTestData()
    y = datareader.readTestLabels()
    accuracy = 0.0

    for i in range(X.shape[0]):
        label = clf.predict(X[i])
        if label == y[i]:
            accuracy += 1.0
    accuracy2 = accuracy/len(y)

    return accuracy1,accuracy2

print 'Naive Bayes'
beforeTime = datetime.now()
nb = NaiveBayesClassifier()
accuracy1,accuracy2 = fitAndPredict(nb)
afterTime = datetime.now()
print 'Time Taken:', afterTime-beforeTime
print 'Accuracy', accuracy1,accuracy2
print '-------------------------------------------------------------------------------------------------------------'
print 'Logistic Regression'
beforeTime = datetime.now()
lr = LogisticRegression(step_size = 0.0001)
accuracy1,accuracy2 = fitAndPredict(lr)
afterTime = datetime.now()
print 'Time Taken:', afterTime-beforeTime
print 'Accuracy', accuracy1,accuracy2
print '-------------------------------------------------------------------------------------------------------------'
