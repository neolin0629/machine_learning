import os
import sys
import tensorflow as tf
import numpy as np

models_path = r'E:\machine-learning\models'
sys.path.append(models_path)

from official.mnist import dataset

# hyper parameters
LEARNING_RATE = 1e-4
TRAINING_EPOCHS = 20
BATCH_SIZE = 100

mnist_train = dataset.train(r"E:\machine-learning\machine_learning\MNIST_data")
mnist_test = dataset.test(r"E:\machine-learning\machine_learning\MNIST_data")

def train_input_fn(features, labels, batch_size):
    pass

def cnn_model_fn(features, labels, mode):
    """
    Input Layer
    Reshape X to 4-D tensor: [batch_size, width, height, channels]
    MNIST images are 28x28 pixels, and have one color channel
    """
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)



    pass

if __name__ == "__main__":
    input_layer = tf.reshape(mnist_train["x"], [-1, 28, 28, 1])
    iterator = input_layer.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))