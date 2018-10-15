# -*- encoding:utf-8 -*-

import os
import numpy
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy import io

class MiniBatch(dict):
    def __init__(self):
        self.adjacent_matrix    = None
        self.label              = None
        self.data               = None

def default_logger(s):
    print(s)

class Graph:
    def __init__(self, edge_path="", node_path="", graph_path="", negative_sample_rate=0.001,logger=None):
        self.negative_sample_rate   = negative_sample_rate
        self.adjacent_matrix        = None
        self.node_number            = 0
        self.edge_number            = 0
        self.order                  = None
        self.epoch_end              = False
        self.start                  = 0
        self.end                    = 0
        self.label                  = None
        if logger == None:
            self.logger = default_logger
        else:
            self.logger = logger
        if (os.path.exists(edge_path) and os.path.exists(node_path)) or \
            os.path.exists(graph_path):
            self.init_variables(edge_path, node_path, graph_path, negative_sample_rate)

    def init_variables(self,edge_path, node_path, graph_path, negative_sample_rate):
        self.negative_sample_rate = negative_sample_rate
        self.logger("Graph Init Begin")
        if os.path.exists(graph_path):
            self.logger("Load Data From Mat")
            self.__load_from_mat(graph_path)
        else:
            self.logger("Load Data From CSV")
            self.__load_from_file(edge_path, node_path)
            self.__save_to_mat(graph_path)
            
        self.logger("%d Nodes In This Graph"%self.node_number)
        self.logger("%d Edges In This Graph"%self.edge_number)
        self.logger("Graph Init Done")
        self.order      = numpy.arange(self.node_number)
        self.epoch_end  = False
        self.start      = 0
        self.end        = 0
        self.label      = None

    def __load_from_file(self, edge_path, node_path):
        with open(node_path,'r') as F:
            L = F.read().split()
            L = [i.strip() for i in L if i.strip()!=""]
            self.node_number = max([int(i) for i in L])
        self.adjacent_matrix = dok_matrix((self.node_number, self.node_number), dtype=int)
        with open(edge_path, 'r') as F:
            L = F.read().split("\n")
            L = [i.strip().split(",") for i in L if i.strip()!=""]
            for x,y in L:
                x = int(x) - 1
                y = int(y) - 1
                self.adjacent_matrix[x,y] = 1
                self.adjacent_matrix[y,x] = 1
        self.edge_number = self.adjacent_matrix.count_nonzero()/2
        self.__ngative_sample()
        self.adjacent_matrix.tocsr()
    
    def __save_to_mat(self, graph_path):
        data = {
            "adjacent_matrix":self.adjacent_matrix,
            "edge_number":self.edge_number,
            "node_number":self.node_number
        }
        io.savemat(graph_path, data)

    def __load_from_mat(self, adj_path):
        data = io.loadmat(adj_path)
        self.adjacent_matrix    = data["adjacent_matrix"]
        self.edge_number        = data["edge_number"][0,0]
        self.node_number        = data["node_number"][0,0]

    def __ngative_sample(self):
        self.logger("Grap Negative Sampling Begin")
        size = 0
        while (size < int(self.negative_sample_rate*self.edge_number)):
            xx = numpy.random.randint(0, self.node_number-1)
            yy = numpy.random.randint(0, self.node_number-1)
            if (xx == yy or self.adjacent_matrix[xx, yy] != 0):
                continue
            self.adjacent_matrix[xx, yy] = -1
            self.adjacent_matrix[yy, xx] = -1
            size += 1
        self.logger("Graph nNegative Sampling Done")

    def sample(self, batch_size, shuffle=True, with_label = False):
        if self.epoch_end:
            if shuffle:
                numpy.random.shuffle(self.order)
            else:
                self.order = numpy.sort(self.order)
            self.start      = 0
            self.epoch_end  = False
        mini_batch                  = MiniBatch()
        self.end                    = int(min(self.node_number, self.start + batch_size))
        index                       = self.order[self.start:self.end]     
        mini_batch.data             = self.adjacent_matrix[index].toarray()
        mini_batch.adjacent_matrix  = self.adjacent_matrix[index].toarray()[:][:,index]
        if with_label and self.label:
            mini_batch.label = self.label[index]
        if (self.end == self.node_number):
            self.end = 0
            self.epoch_end = True
        self.start = self.end
        return mini_batch
    
    def load_lable(self, label_path):
        self.label = numpy.zeros(self.node_number) -1
        with open(label_path,'r') as F:
            S = F.read().split("\n")
            L = [i.strip() for i in S if i!=""]
            L = [i.split(",") for i in L]
            for x,y in L:
                x = int(x) - 1
                y = int(y) - 1
                self.label[x] = y
    
    def subgraph(self, method="link", sample_rate = 0.01):
        if method == "link":
            return self.subgraph_link(sample_rate)
        elif method == "node":
            return self.subgraph_node(sample_rate)
        else:
            return self.subgraph_explore(sample_rate)

    def subgraph_link(self, sample_rate):
        pass
    def subgraph_node(self, sample_rate):
        pass
    def subgraph_explore(self, sample_rate):
        pass
    def draw(self):
        pass



