__author__ = 'santhosh'

import csv
import numpy

def read_data():
    with open('../bupa.data','r') as f:
        csv_data = csv.reader(f, delimiter=',')
        data = []
        for row in csv_data:
            data.append(row)
        data = numpy.array(data)
        return data

