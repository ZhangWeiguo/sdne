import sys
from basic.logger import Logger
from config import Config
from model.sdne import SDNE
from graph.graph import Graph

if __name__ == "__main__":
    logger = Logger("SDNE","sdne.log","None")
    logger.info("Config Begin")
    config = Config()
    config.parse_from_inifile("config/Blogcatalog.ini")
    logger.info("Config Done")

    edge_path   = config.edge_path
    ndoe_path   = config.node_path
    graph_path  = config.graph_path
    batch_size  = config.batch_size

    logger.info("Graph Construct Begin")
    graph = Graph(  edge_path=edge_path, 
                    node_path=ndoe_path, 
                    graph_path=graph_path,
                    logger=logger.info)
    logger.info("Graph Construct Done")

    if config.struct[0] == -1:
        config.struct[0] = graph.node_number

    print(config.struct)
    model = SDNE(config, logger=logger.info)
    # model.rbm_init(graph)
    model.train(graph)
