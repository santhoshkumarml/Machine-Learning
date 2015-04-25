__author__ = 'santhosh'

import numpy
import scipy
import matplotlib.pyplot as plt
from scipy.sparse import linalg

class Graph():
    def __init__(self):
        self.nodes = dict()
        self.edges = dict()

    def add_node(self, node):
        self.nodes[node] = node

    def add_edge(self, node1, node2):
        self.edges[(node1, node2)] = 1
        self.edges[(node2, node1)] = 1

    def getDegreeMatrix(self):
        D = numpy.zeros(shape=(len(self.nodes), len(self.nodes)), dtype=int)
        adj_matrix = self.get_adjacency_matrix()
        for idx in range(len(self.nodes)):
            D[idx][idx] = sum(adj_matrix[idx])
        return D

    def get_adjacency_matrix(self):
        adj_matrix = numpy.zeros(shape=(len(self.nodes), len(self.nodes)), dtype=int)
        for n1_idx in range(len(self.nodes)):
            node1 = self.nodes[n1_idx+1]
            for n2_idx in range(len(self.nodes)):
                node2 = self.nodes[n2_idx+1]
                if node1 != node2:
                    if (node1, node2) in self.edges:
                        adj_matrix[n1_idx][n2_idx] = 1
        return adj_matrix

    def getUnnormalizedLaplacianMatrix(self):
        D = self.getDegreeMatrix()
        W = self.get_adjacency_matrix()
        return numpy.subtract(D, W)


class SpectralClustering:
    def __init__(self, graph):
        self.graph = graph

    def findEigenValuesAndEigenVectors(self, top):
        laplacian = self.graph.getUnnormalizedLaplacianMatrix()
        # eig_va, eig_vector = numpy.linalg.eig(laplacian)
        # sorted_order = sorted(range(10), key= lambda key: eig_va[key])
        A = numpy.ndarray.astype(laplacian, dtype='float')
        eig_va, eig_vector = numpy.linalg.eig(laplacian)
        sorted_order = sorted(range(10), key= lambda key: eig_va[key])
        result = []
        for i in range(10):
            idx = sorted_order[i]
            result.append((eig_va[idx], eig_vector[idx]))
        return result

def plotEigenValue(eigen_vector):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.title('Node Vs Eigen Vector')
    plt.xlabel('Node')
    plt.ylabel('Eigen Vector')
    ax1.plot(range(1, 11), eigen_vector)
    plt.show()


nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
edges = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5), \
         (6, 7), (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10), \
         (1, 6)]

graph = Graph()
for node in nodes:
    graph.add_node(node)
for edge in edges:
    node1, node2 = edge
    graph.add_edge(node1, node2)

spc = SpectralClustering(graph)
result = spc.findEigenValuesAndEigenVectors(2)
for eig_val, eig_vector in result:
    plotEigenValue(eig_vector)

# from sklearn.cluster.spectral import spectral_clustering as spcl
# labels = spcl(graph.get_adjacency_matrix(),n_clusters=2)
# print labels