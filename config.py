# -*- encoding:utf-8 -*-
import sys
from basic.config_parse import IniConfiger


class Config:
    def __init__(self):
        self.gpu                = False
        self.struct             = [-1, 1000, 100]
        self.sparse             = True
        self.dbn_batch_size     = 10
        self.dbn_learning_rate  = 0.1
        self.dbn_epochs         = 10
        self.batch_size         = 10
        self.epochs             = 10
        self.learning_rate      = 0.1
        self.alpha              = 100
        self.beta               = 10
        self.gamma              = 1
        self.reg                = 1
        self.model_path         = "model"
        self.embedding_path     = "embdding"
        self.node_path          = "node"
        self.edge_path          = "edge"
        self.label_path         = "label"
        self.graph_path         = "mat"
    
    def parse_from_inifile(self, ini_path):
        config = IniConfiger(ini_path)
        self.gpu                = bool(config.get("sys","gpu","int"))
        self.sparse             = bool(config.get("sys","sparse","int"))

        self.struct             = list(map(int,config.get("model","struct").split(",")))
        self.dbn_batch_size     = config.get("model","dbn_batch_size","int")
        self.dbn_learning_rate  = config.get("model","dbn_learning_rate","float")
        self.dbn_epochs         = config.get("model","dbn_epochs","int")
        self.batch_size         = config.get("model","batch_size","int")
        self.epochs             = config.get("model","epochs","int")
        self.learning_rate      = config.get("model","learning_rate","float")
        self.alpha              = config.get("model","alpha","float")
        self.beta               = config.get("model","beta","float")
        self.gamma              = config.get("model","gamma","float")
        self.reg                = config.get("model","reg","float")

        self.model_path         = config.get("path","model_path")
        self.embedding_path     = config.get("path","embedding_path")
        self.node_path          = config.get("path","node_path")
        self.edge_path          = config.get("path","edge_path")
        self.label_path         = config.get("path","label_path")
        self.graph_path         = config.get("path","graph_path")

