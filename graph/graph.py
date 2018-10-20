# -*- encoding:utf-8 -*-

import os
import numpy
import random
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy import io
from matplotlib import pyplot

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
        self.label                  = None
        if logger == None:
            self.logger = default_logger
        else:
            self.logger = logger
        if (os.path.exists(edge_path) and os.path.exists(node_path)) or \
            os.path.exists(graph_path):
            self.init_variables(edge_path, node_path, graph_path, negative_sample_rate)
        self.color = [
            "aliceblue","antiquewhite","aqua","aquamarine","azure","beige","bisque","black","blanchedalmond",
            "blue","blueviolet","brown","burlywood","cadetblue","chartreuse","chocolate","coral","cornflowerblue",
            "cornsilk","crimson","cyan","darkblue","darkcyan","darkgoldenrod",
            "darkgray","darkgreen","darkkhaki","darkmagenta","darkolivegreen","darkorange",
            "darkorchid","darkred","darksalmon","darkseagreen","darkslateblue","darkslategray",
            "darkturquoise","darkviolet","deeppink","deepskyblue","dimgray","dodgerblue",
            "firebrick","floralwhite","forestgreen","fuchsia","gainsboro","ghostwhite",
            "gold","goldenrod","gray","green","greenyellow","honeydew","hotpink","indianred","indigo","ivory",
            "khaki","lavender","lavenderblush","lawngreen","lemonchiffon","lightblue",
            "lightcoral","lightcyan","lightgoldenrodyellow","lightgreen","lightgray","lightpink",
            "lightsalmon","lightseagreen","lightskyblue","lightslategray","lightsteelblue","lightyellow",
            "lime","limegreen","linen","magenta","maroon","mediumaquamarine",
            "mediumblue","mediumorchid","mediumpurple","mediumseagreen","mediumslateblue","mediumspringgreen",
            "mediumturquoise","mediumvioletred","midnightblue","mintcream","mistyrose",
            "moccasin","navajowhite","navy","oldlace","olive",
            "olivedrab","orange","orangered","orchid","palegoldenrod","palegreen",
            "paleturquoise","palevioletred","papayawhip","peachpuff","peru","pink","plum","powderblue",
            "purple","red","rosybrown","royalblue","saddlebrown","salmon","sandybrown","seagreen",
            "seashell","sienna","silver","skyblue","slateblue","slategray","snow","springgreen",
            "steelblue","tan","teal","thistle","tomato","turquoise","violet","wheat",
            "white","whitesmoke","yellow","yellowgreen"]
        

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

    def __negative_sample(self):
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
        index                       = self.subgraph(index = self.start, size = batch_size)
        mini_batch.data             = self.adjacent_matrix[index].toarray()
        mini_batch.adjacent_matrix  = self.adjacent_matrix[index].toarray()[:][:,index]
        if with_label and self.label:
            mini_batch.label = self.label[index]
        if (self.start >= self.node_number-1):
            self.epoch_end = True
            self.start = 0
        else:
            self.start += 1
        return mini_batch

    def sample_without_repeat(self, batch_size, shuffle=True, with_label = False):
        if self.epoch_end:
            if shuffle:
                numpy.random.shuffle(self.order)
            else:
                self.order = numpy.sort(self.order)
            self.start      = 0
            self.epoch_end  = False
        mini_batch                  = MiniBatch()
        end = min(self.start + batch_size,self.node_number)
        index                       = numpy.arange(self.start, end)
        mini_batch.data             = self.adjacent_matrix[index].toarray()
        mini_batch.adjacent_matrix  = self.adjacent_matrix[index].toarray()[:][:,index]
        if with_label and self.label:
            mini_batch.label = self.label[index]
        if (self.start >= self.node_number-1):
            self.epoch_end = True
            self.start = 0
        else:
            self.start = end
        return mini_batch


    def load_label(self, label_path):
        self.label = numpy.zeros(self.node_number) -1
        with open(label_path,'r') as F:
            S = F.read().split("\n")
            L = [i.strip() for i in S if i!=""]
            L = [i.split(",") for i in L]
            for x,y in L:
                x = int(x) - 1
                y = int(y) - 1
                self.label[x] = y
    
    def subgraph(self, index, size):
        t = [index]
        while True:
            t = self.adjacent_matrix[t,:].nonzero()[1]
            if t.shape[0] >= size:
                numpy.random.shuffle(t)
                t = t[0:size]
                break
        return t


    
    def draw(self, embedding_path, img_path):
        embedding = io.loadmat(embedding_path)["embedding"]
        classes = numpy.unique(self.label)
        classes_number = classes.shape[0]
        color = []
        figure=pyplot.figure()
        for i in range(classes_number):
            index = numpy.argwhere(self.label==classes[i])
            embedding_sub = embedding[index.reshape((-1,))]
            pyplot.scatter(embedding_sub[:,0], embedding_sub[:,1],c=random.choice(self.color))

        min_embedding = numpy.min(embedding)
        max_embedding = numpy.max(embedding)
        if min_embedding < 0:
            min_embedding = min_embedding * 1.2
        else:
            min_embedding = min_embedding * 0.8
        if max_embedding < 0:
            max_embedding = max_embedding * 0.8
        else:
            max_embedding = max_embedding * 1.2
        pyplot.xlim(min_embedding, max_embedding)
        pyplot.ylim(min_embedding, max_embedding)
        figure.savefig(img_path,dpi=500)



