'''
Created on Feb 18, 2015

@author: santhosh
'''
import numpy
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#Y     X1    X2    Count
#+     T     T     3
#+     T     F     5
#+     F     T     5
#+     F     F     2
#-     T     T     0
#-     T     F     2
#-     F     T     3
#-     F     F     5
data = numpy.zeros(shape = (25,3), dtype = bool)
print data.dtype
idx = 0
for i in range(3):
    data[idx]= numpy.array([True,True,True])
    idx+=1
for i in range(5):
    data[idx]= numpy.array([True,False,True])
    idx+=1
for i in range(5):
    data[idx]= numpy.array([False,True,True])
    idx+=1
for i in range(2):
    data[idx]= numpy.array([False,False,True])
    idx+=1

for i in range(0):
    data[idx]= numpy.array([True,True,False])
    idx+=1
for i in range(2):
    data[idx]= numpy.array([True,False,False])
    idx+=1
for i in range(3):
    data[idx]= numpy.array([False,True,False])
    idx+=1
for i in range(5):
    data[idx]= numpy.array([False,False,False])
    idx+=1

X = data[:,[0,1]]
y = data[:,[2]]
dtcl = DecisionTreeClassifier()
dtcl.fit(X, y)
with open("/home/santhosh/tree.dot", 'w') as f:
    f = tree.export_graphviz(dtcl, out_file=f)