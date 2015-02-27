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
    X = datareader.readTestData()
    y = datareader.readTestLabels()
    accuracy = 0.0
    for i in range(X.shape[0]):
        label = clf.predict(X[i])
        if label == y[i]:
            accuracy += 1.0
    return accuracy

print 'Naive Bayes'
beforeTime = datetime.now()
nb = NaiveBayesClassifier()
accuracy = fitAndPredict(nb)
afterTime = datetime.now()
print 'Time Taken:', afterTime-beforeTime
print 'Accuracy', accuracy
print '-------------------------------------------------------------------------------------------------------------'
print 'Logistic Regression'
beforeTime = datetime.now()
lr = LogisticRegression(step_size = 0.0001)
accuracy = fitAndPredict(lr)
afterTime = datetime.now()
print 'Time Taken:', afterTime-beforeTime
print 'Accuracy', accuracy
print '-------------------------------------------------------------------------------------------------------------'
