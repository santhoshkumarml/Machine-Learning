'''
Created on Feb 20, 2015

@author: santhosh
'''
from util import datareader
from naivebayes.NaiveBayes import NaiveBayesClassifier
from logisticregression.LogisticRegression import LogisticRegression


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
    print accuracy/len(y)
    
nb = NaiveBayesClassifier()
lr = LogisticRegression(step_size = 0.0001)
fitAndPredict(lr)