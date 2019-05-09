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
# with tf.variable_scope("hidden_layer") as scope:
#     w = tf.constant([[1., 1.],[1., 1.]], shape=[2, 2])
#     b = tf.constant([0., -1.], shape=[2,])
#     x_ = tf.nn.relu(tf.add(tf.matmul(x, w), b))

# with tf.variable_scope("output") as scope:
#     w = tf.constant([[1.], [-2.]], shape=[2, 1])
#     b = tf.constant([0.], shape=[1,])
#     y_ = tf.add(tf.matmul(x_, w), b)
    
# loss = tf.reduce_mean(tf.square(y - y_))

# with tf.Session() as sess:
#     pred, l = sess.run([y_, loss], feed_dict={x: XOR_X, y: XOR_Y})
#     print(f"XOR problem:  mse={l}, y prediction: \n{pred}")      


# 2. use neural network to optimize paramters
# extend the width of hidden layer to 3
with tf.variable_scope("layer1") as scope:
    w1 = tf.get_variable('w1', shape=[2, 3])
    b1 = tf.get_variable('b1', shape=[3,])
    x1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

with tf.variable_scope("layer2") as scope:
    w2 = tf.get_variable('w2', shape=[3, 1])
    b2 = tf.get_variable('b2', shape=[1,])
    x2 = tf.add(tf.matmul(x1, w2), b2)
   
preds = tf.nn.sigmoid(x2)
init = tf.global_variables_initializer()
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x2))

# define optimizer
learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)
    sess.run(init)    

    for step in range(10000):
        if step < 3000:
            lr = 1
        elif step < 6000:
            lr = 0.1
        else:
            lr = 0.01
        _, pred, l = sess.run([train_step, preds, loss], feed_dict={x: XOR_X, y: XOR_Y, learning_rate: lr})
        if not step % 2000:
            print('Step:{} -> Loss:{} -> Predictions \n{}'.format(step, l, pred))
     