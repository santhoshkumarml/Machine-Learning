__author__ = 'santhosh'

import numpy
import matplotlib.pyplot as plt

class Graph():
    def __init__(self):
        self.nodes = dict()
        self.edges = dict()

    def get_nodes(self):
        return self.nodes.values()

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


def plotEigenValue(eigen_vector):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    plt.title('Node Vs Eigen Vector')
    plt.xlabel('Node')
    plt.ylabel('Eigen Vector')
    ax1.plot(range(1, 11), eigen_vector)
    plt.savefig('Spectral Clustering - EigenVector.png')
    plt.close()


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

print 'Adjacency Matrix:'
print graph.get_adjacency_matrix()
print 'Unnormalized Graph Laplacian:'
print graph.getUnnormalizedLaplacianMatrix()

n_clusters = 2
n_nodes = len(graph.get_nodes())

laplacian = graph.getUnnormalizedLaplacianMatrix()
A = numpy.ndarray.astype(laplacian, dtype='float')
eig_vals, eig_vectors = numpy.linalg.eigh(A)

second_eig_vector = eig_vectors[:,1]
plotEigenValue(second_eig_vector)


# for col in range(10):
#     eig_val = eig_vals[col]
#     eig_vector = eig_vectors[:,col]
#     print eig_val, eig_vector
#     plotEigenValue(eig_vector)

