from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# a = tf.constant(3.0, dtype=tf.float32)
# b = tf.constant(4.0) 
# total = a + b
# writer = tf.summary.FileWriter('.')
# writer.add_graph(tf.get_default_graph())
# writer.flush()

sess = tf.Session()
# print(sess.run({'ab':(a, b), 'total':total}))
# x = tf.placeholder(tf.float32, shape=[None, 3])
# linear_model = tf.layers.Dense(units=1)
# y = linear_model(x)

# x = tf.constant([[1],[2],[3],[4]], dtype=tf.float32)
# y_true = tf.constant([[0],[-1],[-2],[-3]], dtype=tf.float32)
# linear_model = tf.layers.Dense(units=1)
# y_pred = linear_model(x)
# loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# init = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init)

# for i in range(100):
#     _, loss_value = sess.run((train, loss))
#     print(loss_value)

# print(sess.run(y_pred))
