from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# XOR DL Chapter 6 
# use tensorflow low-level api
# 1 hidden layer, 2 nodes, use relu

x = tf.placeholder(tf.float32, shape=[4,2], name='x')
y = tf.placeholder(tf.float32, shape=[4,1], name='y')

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]

# 1. with no parameters
with tf.variable_scope("hidden_layer") as scope:
    w1 = tf.constant([[1, 1],[1, 1]], shape=[2, 2], dtype=tf.float32)
    b1 = tf.constant([0, -1], shape=[2,], dtype=tf.float32)
    x_ = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

with tf.variable_scope("output") as scope:
    w2 = tf.constant([[1], [-2]], shape=[2, 1], dtype=tf.float32)
    b2 = tf.constant([0], shape=[1,], dtype=tf.float32)
    y_ = tf.add(tf.matmul(x_, w2), b2)
    
loss = tf.reduce_mean(tf.square(y - y_))
# init = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(init)
    x, pred, l = sess.run([x_, y_, loss], feed_dict={x: XOR_X, y: XOR_Y})
    print(f"XOR problem:  mse={l}, y prediction: \n{pred}")      


# 2. 

# w1 = tf.Variable(tf.random_uniform([2,2], -1, 1))
# w2 = tf.Variable(tf.random_uniform([2,1], -1, 1))

# b1 = tf.Variable(tf.zeros([2]))
# b2 = tf.Variable(tf.zeros([1]))



# # build a network
# h = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
# y_ = tf.nn.sigmoid(tf.add(tf.matmul(h, w2), b2))

# # define loss function(MSE)

# # define optimizer
# learning_rate = tf.placeholder(tf.float32)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# sess = tf.Session()

# writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

# sess.run(init)
# for step in range(10000):
#     if step < 3000:
#         lr = 1
#     elif step < 6000:
#         lr = 0.1
#     else:
#         lr = 0.01
#     _, pred, l = sess.run([train_step, y_, loss], feed_dict={x: XOR_X, y: XOR_Y, learning_rate: lr})
#     if step % 500 == 0:
#         print('Epoch ', step)
#         print('y_ ', pred)
#         print('w1 ', sess.run(w1))
#         print('b1 ', sess.run(b1))
#         print('w2 ', sess.run(w1))
#         print('b2 ', sess.run(b2))
#         print('loss ', l)