'''
Created on Feb 20, 2015

@author: santhosh
'''


from naivebayes.NaiveBayes import NaiveBayesClassifier
import numpy

train_data = '../train.data'
train_target = '../train.label'

data = []
y = []

with open(train_data) as f:
    lines = f.readlines()
    for line in lines:
        doc_id, word_id, count = line.split()
        doc_id = int(doc_id)
        word_id = int(word_id)
        count = int(count)
        data.append([doc_id, word_id, count])

data = numpy.array(data)

with open(train_target) as f:
    lines = f.readlines()
    for line in lines:
        classified_as = line.strip()
        y.append(int(classified_as))

max_doc_id = max(data[:,0])
max_word_id = max(data[:,1])

X = numpy.array(shape = (max_doc_id, max_word_id), dtype = int)
for doc_id, word_id, count in data:
    X[doc_id-1][word_id-1] = count
y = numpy.array(y)

nb = NaiveBayesClassifier()
nb.fit()
