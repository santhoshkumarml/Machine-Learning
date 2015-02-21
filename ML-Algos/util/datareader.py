'''
Created on Feb 18, 2015

@author: santhosh
'''

import numpy
train_data = '../train.data'
train_target = '../train.label'
test_data = '../test.data'
test_target = '../test.label'

def read_data(data_file):
    data = []
    with open(data_file) as f:
        lines = f.readlines()
        for line in lines:
            doc_id, word_id, count = line.split()
            doc_id = int(doc_id)
            word_id = int(word_id)
            count = int(count)
            data.append([doc_id, word_id, count])
    return data

def read_label(label_file):
    y = []
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            classified_as = line.strip()
            y.append(int(classified_as))
            
    return y

def compactData(data):
    data = numpy.array(data)
    max_doc_id = max(data[:,0])
    max_word_id = max(data[:,1])
    X = numpy.zeros(shape = (max_doc_id, max_word_id), dtype = int)
    
    for doc_id, word_id, count in data:
        X[doc_id-1][word_id-1] = count
    return X

def readTrainDataAndLabels():
    data = read_data(train_data)
    labels = read_label(train_target)
    
    X = compactData(data)
    y = numpy.array(labels)
    
    return X,y

def readTestData():
    data = read_data(test_data)
    X = compactData(data)
    return X

def readTestLabels():
    labels = read_label(test_target)
    y = numpy.array(labels)
    return y