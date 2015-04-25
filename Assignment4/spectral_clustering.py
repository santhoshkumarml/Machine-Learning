__author__ = 'santhosh'

import numpy

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
        D = numpy.zeros(len(self.nodes), len(self.nodes))
        adj_matrix = self.get_adjacency_matrix()
        for idx in len(self.nodes):
            D[idx] = sum(adj_matrix[idx])
        return D

    def get_adjacency_matrix(self):
        adj_matrix = numpy.zeros(len(self.nodes), len(self.nodes))
        for n1_idx in len(self.nodes):
            node1 = self.nodes[n1_idx]
            for n2_idx in len(self.nodes):
                node2 = self.nodes[n2_idx]
                if node1 != node2:
                    if (node1, node2) in self.edges:
                        adj_matrix[n1_idx][n2_idx] = 1
        return adj_matrix

    def createUnnormalizedLaplacianMatrix(self):
        D = self.getDegreeMatrix()
        W = self.get_adjacency_matrix()
        return numpy.subtract(D,W)


class SpectralClustering:
    def __init__(self, graph):
        self.graph = graph

    def findEigenValuesAndEigenVectors(self, topVectors):
        laplacian = self.graph.createUnnormalizedLaplacianMatrix()
        eig_va, eig_vector = numpy.linalg.eig(laplacian)