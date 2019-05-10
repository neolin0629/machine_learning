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
    w = tf.constant([[1., 1.],[1., 1.]], shape=[2, 2])
    b = tf.constant([0., -1.], shape=[2,])
    x_ = tf.nn.relu(tf.add(tf.matmul(x, w), b))

with tf.variable_scope("output") as scope:
    w = tf.constant([[1.], [-2.]], shape=[2, 1])
    b = tf.constant([0.], shape=[1,])
    y_ = tf.add(tf.matmul(x_, w), b)
    
loss = tf.reduce_mean(tf.square(y - y_))

with tf.Session() as sess:
    pred, l = sess.run([y_, loss], feed_dict={x: XOR_X, y: XOR_Y})
    print(f"XOR problem:  mse={l}, y prediction: \n{pred}")      