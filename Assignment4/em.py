__author__ = 'santhosh'

import numpy
import math

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
        c_p_1 = self.thetas[0]*math.exp(-(((self.data[train_ins_idx] - self.means[0])**2)/2))
        c_p_2 = self.thetas[1]*math.exp(-(((self.data[train_ins_idx] - self.means[1])**2)/2))
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
            print current_likelihood, self.max_likelihood
            if current_likelihood < self.max_likelihood:
                break
            self.max_likelihood = current_likelihood
            self.means, self.thetas = self.M_STEP()
        return self.means, self.thetas


with open('hw5.data.txt') as f:
    data = [float(line.strip()) for line in f.readlines()]
means = (1,2)
thetas = (0.33,0.67)
em = EM(means, thetas, data)
print em.EM_ALGO()