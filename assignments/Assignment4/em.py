__author__ = 'santhosh'

import numpy
import math
from matplotlib import mlab
from matplotlib import pyplot as plt

class EM:
    def __init__(self, means, thetas, data):
        self.means = means
        self.thetas = thetas
        self.max_likelihood = -float('inf')
        self.class_conditional_probs = []
        self.data = data

    def prob_x(self, train_ins_idx, class_type):
        mean = self.means[class_type]
        theta = self.thetas[class_type]
        prob = theta*math.exp(-(((self.data[train_ins_idx] - mean)**2)/2))
        return prob

    def calculateClassConditionalProbability(self, train_ins_idx):
        c_p_1 = self.thetas[0]*(1/math.sqrt(2*math.pi))*math.exp(-(((self.data[train_ins_idx] - self.means[0])**2)/2))
        c_p_2 = self.thetas[1]*(1/math.sqrt(2*math.pi))*math.exp(-(((self.data[train_ins_idx] - self.means[1])**2)/2))
        denom = c_p_1+c_p_2
        return c_p_1/denom, c_p_2/denom

    def E_STEP(self):
        self.class_conditional_probs = numpy.zeros(shape=(2, len(self.data)), dtype='float')
        current_exp_likelihood = 0
        for j in range(len(self.data)):
            c_p_1, c_p_2 = self.calculateClassConditionalProbability(j)
            self.class_conditional_probs[0][j] = c_p_1
            self.class_conditional_probs[1][j] = c_p_2
            current_exp_likelihood += c_p_1*self.prob_x(j, 0)+c_p_2*self.prob_x(j, 1)
        return current_exp_likelihood

    def M_STEP(self):
        mean1, mean2 = 0.0, 0.0
        theta1, theta2 = 0.0, 0.0
        for i in range(len(self.data)):
            mean1 += self.class_conditional_probs[0][i]*self.data[i]
            mean2 += self.class_conditional_probs[1][i]*self.data[i]
            theta1 += self.class_conditional_probs[0][i]
            theta2 += self.class_conditional_probs[1][i]
        mean1 = mean1/theta1
        mean2 = mean2/theta2
        denom = theta1+theta2
        theta1 /= denom
        theta2 /= denom
        return (mean1, mean2), (theta1, theta2)

    def EM_ALGO(self):
        while True:
            current_likelihood = self.E_STEP();
            # print current_likelihood, self.max_likelihood
            if current_likelihood < self.max_likelihood:
                break
            self.max_likelihood = current_likelihood
            self.means, self.thetas = self.M_STEP()
        return self.means, self.thetas


def runEM(data):
    means = (1, 2)
    thetas = (0.33, 0.67)
    em = EM(means, thetas, data)
    (mean1, mean2), (theta1, theta2) = em.EM_ALGO()
    print 'Means:', mean1, mean2
    print 'Thetas:', theta1, theta2
    return (mean1, mean2), (theta1, theta2)

def plotHistoGramAndDensity(data, mean1, mean2, theta1, theta2):
    plt.hist(data, range=[-5, 6])
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig('Historgram.png')
    plt.close()

    data1 = [i for i in numpy.arange(-5, 6, 0.0011)] #same as number of data points

    norm1 = mlab.normpdf(data1, mean1, 1)
    norm2 = mlab.normpdf(data1, mean2, 1)

    density = [0 for i in range(len(data1))]
    for i in range(len(data1)):
        density[i] = norm1[i] * theta1 + norm2[i] * theta2

    plt.title("Density")
    plt.ylabel("Frequency")
    plt.plot(data1, density)
    plt.savefig('Density.png')
    plt.close()


def plotContour(data, thetas):
    x = [i for i in numpy.arange(-1, 4, 0.25)]
    y = [i for i in numpy.arange(-1, 4, 0.25)]
    size = len(x)
    z = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            em = EM((x[i], y[j]), thetas, data)
            likelihood = em.E_STEP()
            z[i][j] = likelihood

    fig = plt.figure()
    CS = plt.contour(x, y, z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Contour Plot of Means and Likelihood')
    plt.savefig('Contour.png')
    plt.close()

with open('hw5.data.txt') as f:
    data = [float(line.strip()) for line in f.readlines()]

(mean1, mean2), (theta1, theta2) = runEM(data)
plotHistoGramAndDensity(data, mean1, mean2, theta1, theta2)
plotContour(data, (theta1, theta2))