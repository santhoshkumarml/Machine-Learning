'''
Created on Feb 20, 2015

@author: santhosh
'''

import numpy

array = numpy.arange(0,1000,1)
array = numpy.reshape(array, (200,5))
print array[:,0]
