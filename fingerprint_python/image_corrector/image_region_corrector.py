from __future__ import print_function
import sys
sys.path.append("/Users/shankaragarwal/LP/LatentFingerPrintCorrection/fingerprint_python")
# sys.path.append("/home/ubuntu/code/fingerprint_python")

import math
import os

from math import ceil, floor

from constants import constants as fpconst
from preprocessor import get_images as get_images
from encoder_cnn import encoder
import tensorflow as tf

import numpy as np

#
from PIL import Image

ERROR_ACCEPTED = tf.constant(0.001)

IMAGE_HEIGHT = fpconst.region_height_size
IMAGE_WIDTH = fpconst.region_width_size
dimensions = 300
num_epochs = 10
learning_rate = 0.5
mini_batch_size = 32


def compute_cost(Y, Y_hat):
    loss = tf.subtract(Y, Y_hat)
    loss = tf.pow(loss, 2)
    return tf.reduce_mean(loss)


def initialize_parameters():
    parameters = {
        0: 20000,
        1: 15000,
        2: 10000
    }
    return parameters


def forward_propagation(X, parameters):
    L1 = tf.layers.Dense(X, parameters[0], tf.nn.relu)
    L2 = tf.layers.Dense(L1, parameters[1], tf.nn.relu)
    L3 = tf.layers.Dense(L2, parameters[2], tf.nn.relu)

    L4 = tf.layers.Dense(L3, IMAGE_HEIGHT * IMAGE_WIDTH, activation="relu")

    L4 = L4.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    return L4


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # To make your "random" minibatches the same as ours
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_mini_batches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_mini_batches * mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[num_complete_mini_batches * mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# TODO
def get_encoded_vector(correct_output, encoder_model):
    return np.random.rand(dimensions, 1)


# TODO
def get_corrected_output(latent_image, vector):
    pass


def get_data_set(location, encoder_model):
    X = []
    Y = []
    for dir in os.listdir(location):
        location_dir = location + "/" + dir + "/"
        orig_name = location_dir + fpconst.ORIGINAL_FILE_NAME
        correct_output = np.load(orig_name)
        encoded_vector = get_encoded_vector(correct_output, encoder_model)

        for file in os.listdir(location_dir):
            if file != fpconst.ORIGINAL_FILE_NAME:
                val = np.load(location_dir + file)
                val = np.reshape(val, [IMAGE_HEIGHT * IMAGE_WIDTH, 1])
                val = np.concatenate(val, encoded_vector, axis=0)
                X.append(val)
                Y.append(correct_output)

    test_size = 1000

    test_indexes = np.random.sample(range(0, len(X)), test_size)
    X_test = []
    Y_test = []

    for index in test_indexes:
        X_test.append(X[index])
        Y_test.append(Y[index])

    for index in test_indexes:
        X.pop(index)

    X_train = X
    Y_train = Y
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


def train(X_train, Y_train, X_test, Y_test):
    m = X_train.shape[0]

    seed = 10
    print_cost = True

    tf_input = tf.placeholder(tf.float32, shape=[IMAGE_HEIGHT * IMAGE_WIDTH + dimensions, 1], name="input_x")
    tf_output = tf.placeholder(tf.float32, shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="output")

    parameters = initialize_parameters()

    Z = forward_propagation(tf_input, parameters)

    cost = compute_cost(Z, tf_output)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    costs = []

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            try:
                mini_batch_cost = 0
                num_mini_batches = int(
                    m / mini_batch_size)  # number of mini_batches of size minibatch_size in the train set
                seed = seed + 1
                mini_batches = random_mini_batches(X_train, Y_train, mini_batch_size, seed)
                for mini_batch in mini_batches:
                    (mini_batch_X, mini_batch_Y) = mini_batch

                    _, cost = sess.run([optimizer, cost], feed_dict={tf_input: mini_batch_X, tf_output: mini_batch_Y})
                    mini_batch_cost += cost / num_mini_batches

                # Print the cost every epoch
                if print_cost == True and epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, mini_batch_cost))
                if print_cost == True and epoch % 1 == 0:
                    costs.append(mini_batch_cost)
            except Exception as e:
                # pass
                print(e)

    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per tens)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

    diff = tf.subtract(Z, tf_output)
    diff = tf.pow(diff, 2)
    val = tf.reduce_mean(diff)
    correct_prediction = tf.less_equal(val, ERROR_ACCEPTED)

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({tf_input: X_train, tf_output: Y_train})
    test_accuracy = accuracy.eval({tf_input: X_test, tf_output: Y_test})
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return parameters
