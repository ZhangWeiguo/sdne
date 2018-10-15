# import tensorflow as tf
# import numpy
# from rbm import RBM
# from sklearn import datasets
# bv = tf.Variable(tf.zeros([1,10]), name = "a")
# x = tf.placeholder(tf.int16,name="x")
# init_op = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.run(init_op)
#     print(session.run(bv))
#     print(session.run(x,feed_dict={x:[3,4,5]}))

# data = datasets.load_digits()
# x = data.data
# print(x.shape)
# input_dim = 64
# output_dim = 10
# learning_rate = 0.5
# batch_size = 5
# D = []
# for i in range(1000):
#     D.append(x[i,:])

# Rbm = RBM([input_dim, output_dim], learning_rate, batch_size)
# for i in range(10):
#     for i in range(100):
#         ret = Rbm.fit(D[i*10:(i+1)*10])
#         print(ret)
# para = Rbm.get_para()
# for i in para:
#     print(i.shape)

