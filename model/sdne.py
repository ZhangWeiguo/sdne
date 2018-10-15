import numpy as np
import tensorflow as tf
import time
import copy
import random
import os
from .rbm import RBM
from scipy import io


'''
RBM可用于预训练参数
'''
def default_logger(s):
    print(s)

class SDNE:
    def __init__(self, config, logger=None):
        '''
        self.layers
        self.struct
        self.sparse
        self.w
        self.b

        Cofnig:
            config.gpu
            config.struct
            config.sparse
            config.dbn_batch_size
            config.dbn_learning_rate
            config.dbn_epochs

            config.batch_size
            config.epochs
            config.learing_rate
            config.alpha
            config.beta
            config.gamma
            config.reg

        Graph:
            data.adjacent_matrix
            data.node_number
            data.sample
        
        MiniBatch
            data.data
            data.adjacent_matrix
            data.label

        '''

        self.config = config
        self.init = False
        if logger != None:
            self.logger = logger
        else:
            self.logger = default_logger
        
        # GPU option
        if self.config.gpu:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config =  tf_config)
        else:
            self.sess = tf.Session()

        self.layers = len(config.struct)
        self.struct = config.struct
        self.sparse = config.sparse
        self.w = {}
        self.b = {}
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.w[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.w[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        self.struct.reverse()


        # input                
        self.adjacent_matrix = tf.placeholder(tf.float32, [None, None])

        # sparse
        self.x_sp_indices   = tf.placeholder(tf.int64)
        self.x_sp_ids_val   = tf.placeholder(tf.float32)
        self.x_sp_shape     = tf.placeholder(tf.int64)
        self.x_sp           = tf.SparseTensor(self.x_sp_indices, self.x_sp_ids_val, self.x_sp_shape)

        self.x = tf.placeholder("float", [None, config.struct[0]])
        

        self.__make_compute_graph()
        self.loss = self.__make_loss()
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()       
        self.sess.run(init)
        

    
    def __make_compute_graph(self):
        def encoder(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                if i == 0 and self.config.sparse:
                    X = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X, self.w[name]) + self.b[name])
                else:
                    X = tf.nn.sigmoid(tf.matmul(X, self.w[name]) + self.b[name])
            return X
        def decoder(X):
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.w[name]) + self.b[name])
            return X
            
        self.h = encoder(self.x)
        self.x_reconstruct = decoder(self.h)
    
    def __make_loss(self):
        def get_1st_loss_link_sample(self, Y1, Y2):
            return tf.reduce_sum(tf.pow(Y1 - Y2, 2))
        def get_1st_loss(H, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch,1))
            L = D - adj_mini_batch
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X)* B, 2))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.itervalues()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
            return ret
            

        self.loss_2nd = get_2nd_loss(self.x, self.x_reconstruct, self.config.beta)
        self.loss_1st = get_1st_loss(self.h, self.adjacent_matrix)
        self.loss_xxx = tf.reduce_sum(tf.pow(self.x_reconstruct,2))
        # we don't need the regularizer term, since we have nagetive sampling.
        # self.loss_reg = get_reg_loss(self.w, self.b) 
        # return config.gamma * self.loss_1st + config.alpha * self.loss_2nd + config.reg * self.loss_reg
        
        return self.config.gamma * self.loss_1st + self.config.alpha * self.loss_2nd +self.loss_xxx

    def save_model(self, path):
        saver = tf.train.Saver(self.b.values() + self.w.values())
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver(self.b.values() + self.w.values())
        saver.restore(self.sess, path)
        self.init = True
    
    def rbm_init(self, data):
        def assign(a, b):
            op = a.assign(b)
            self.sess.run(op)
        self.logger("Init Para By RBM Model Begin")
        if os.path.exists(self.config.model_path):
            self.restore_model(self.config.model_path)
            self.logger("Restore Model From " + self.config.model_path)
        else:
            shape = self.struct
            rbms = []
            for i in range(self.layers - 1):
                rbm_unit = RBM(
                    shape           = [shape[i], shape[i+1]], 
                    batch_size      = self.config.dbn_batch_size, 
                    learning_rate   = self.config.dbn_learning_rate)
                rbms.append(rbm_unit)
                for epoch in range(self.config.dbn_epochs):
                    error = 0
                    for batch in range(0, data.node_number, self.config.dbn_batch_size):
                        mini_batch = data.sample(self.config.dbn_batch_size).data
                        for k in range(len(rbms) - 1):
                            mini_batch = rbms[k].predict(mini_batch)
                        error += rbm_unit.fit(mini_batch)
                    self.logger("%d Layer: Rbm Epochs %3d Error: %5.6f"%(i,epoch,error))

                W, bv, bh = rbm_unit.get_para()
                name = "encoder" + str(i)
                assign(self.w[name], W)
                assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                assign(self.w[name], W.transpose())
                assign(self.b[name], bv)
        self.init = True
        self.logger("Init Para By RBM Model Done")

    def __get_feed_dict(self, data):
        x               = data.data
        adjacent_matrix = data.adjacent_matrix
        if self.sparse:
            x_ind   = np.vstack(np.where(x)).astype(np.int64).T
            x_shape = np.array(x.shape).astype(np.int64)
            x_val   = x[np.where(x)]
            return {self.x                  :   x, 
                    self.x_sp_indices       :   x_ind, 
                    self.x_sp_shape         :   x_shape, 
                    self.x_sp_ids_val       :   x_val,
                    self.adjacent_matrix    :   adjacent_matrix}
        else:
            return {self.x: x, self.adjacent_matrix: adjacent_matrix}


    # train all data by batch
    def train(self, graph):
        batch_size       = self.config.batch_size
        model_path       = self.config.model_path
        embedding_path   = self.config.embedding_path
        current_epoch    = 0
        while True:
            if current_epoch <= self.config.epochs:
                current_epoch += 1
                loss = 0
                embedding = None
                while not graph.epoch_end:
                    mini_batch = graph.sample(batch_size)
                    loss = self.fit(mini_batch)
                    self.logger("SDNE Epoch %3d Error: %5d/%5d %5.6s"%(current_epoch, graph.start, graph.node_number, loss))
                model_path_epoch = model_path + ".%s"%str(current_epoch)
                embedding_path_epoch = embedding_path + ".%s"%str(current_epoch)
                self.save_model(model_path_epoch)

                while not graph.epoch_end:
                    mini_batch = graph.sample(batch_size, shuffle=False)
                    loss += self.get_loss(mini_batch)
                    if embedding is None:
                        embedding = self.get_embedding(mini_batch)
                    else:
                        embedding = np.vstack((embedding, self.get_embedding(mini_batch)))
                io.savemat(embedding_path_epoch, embedding)
                self.logger("SDNE Epoch %3d Error: %5.6s"%(current_epoch, loss))
            else:
                break











    # train one batch
    def fit(self, data):
        feed_dict = self.__get_feed_dict(data)
        ret, _ = self.sess.run((self.loss, self.optimizer), feed_dict = feed_dict)
        return ret
    
    def get_loss(self, data):
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.h, feed_dict = self.__get_feed_dict(data))

    def get_w(self):
        return self.sess.run(self.w)
        
    def get_b(self):
        return self.sess.run(self.b)
        
    def close(self):
        self.sess.close()

    

