'''
Created on Feb 20, 2015

@author: santhosh
'''
from util import datareader
from naivebayes.NaiveBayes import NaiveBayesClassifier
from logisticregression.LogisticRegression import LogisticRegression

nb = NaiveBayesClassifier()

lr = LogisticRegression(step_size=0.0001)
X,y = datareader.readTrainDataAndLabels()

#nb.fit(X, y)
lr.fit(X, y)

X = datareader.readTestData()
y = datareader.readTestLabels()

accuracy = 0.0

for i in range(X.shape[0]):
    label = lr.predict(X[i])
    if label == y[i]:
        accuracy += 1.0

print accuracy/len(y)
