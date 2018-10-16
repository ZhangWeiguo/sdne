import numpy
from matplotlib import pyplot
from graph.graph import Graph
from config import Config
from scipy import io

def cal_precision_k():
    pass
def cal_ap():
    pass
def cal_map():
    pass
def cal_macro_f1():
    pass
def cal_micro_f1():
    pass



config = Config()
config.parse_from_inifile("config/Blogcatalog.ini")
edge_path   = config.edge_path
ndoe_path   = config.node_path
graph_path  = config.graph_path
batch_size  = config.batch_size


graph = Graph(  edge_path=edge_path, 
                node_path=ndoe_path, 
                graph_path=graph_path)
graph.load_label(config.label_path)

graph.draw("data\\BlogCatalog\\data\\2d\\embedding.mat.50","groups.jpg")





# n = 0
# D = {}
# with open(config.label_path,'r') as F:
#     S = F.read()
#     L = S.split("\n")
#     for line in L:
#         line = line.strip()
#         x,y = line.split(",")
#         x = int(x)
#         y = int(y)
#         if x in D:
#             n += 1
#         else:
#             D[x] = y
# print(n,len(D))


