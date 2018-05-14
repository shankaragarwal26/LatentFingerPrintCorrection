'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import math
import sys

#
# sys.path.append("/Users/shankaragarwal/LatentFingerprint/fingerprint_python")
sys.path.append("/home/ubuntu/code/fingerprint_python")

import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

import tensorflow as tf

# import matplotlib.pyplot as plt

from constants import constants as fpconst

from scipy import misc
from PIL import Image

batch_size = 32
num_classes = 10
epochs = 12

# Tensor Flow Properties
dimensions = 300
learning_rate = 0.2
num_epochs = 12
mini_batch_size = 8
alpha = tf.constant(0.8)
beta = tf.constant(0.7)


def get_region_from_file_name(file):
    names = file.split(".")[0]
    names = names.split("_")
    return int(names[len(names) - 1])


def read_data(location):
    result = []
    for file in os.listdir(location):
        if not file.startswith("."):
            region = get_region_from_file_name(file)
            arr = [np.load(location + file), region]
            result.append(arr)

    return result


def divide_into_train_test(full_data_set):
    size = len(full_data_set)
    train_size = 0.8 * size
    idx = np.random.choice(np.arange(size), int(train_size), replace=False)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(size):
        if i in idx:
            x_train.append(full_data_set[i][0])
            y_train.append(full_data_set[i][1])
        else:
            x_test.append(full_data_set[i][0])
            y_test.append(full_data_set[i][1])

    return x_train, y_train, x_test, y_test


def get_data(loc):
    full_data_set = read_data(loc)

    x_train, y_train, x_test, y_test = divide_into_train_test(full_data_set)

    print(len(x_train))
    print(len(x_test))
    print(len(y_train))
    print(len(y_test))

    print(x_train[0])
    print(y_train[0])

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x_test.reshape((x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))

    y_train = keras.utils.to_categorical(y_train, fpconst.regions)
    y_test = keras.utils.to_categorical(y_test, fpconst.regions)

    input_shape = (fpconst.region_height_size, fpconst.region_width_size, 1)

    return x_train, y_train, x_test, y_test, input_shape


def get_model(x_train, y_train, x_test, y_test, input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    # Layer 1
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layer2
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Layer3
    model.add(Conv2D(256, (2, 2), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # # Layer4
    # model.add(Conv2D(512, (2, 2), strides=(2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    # FC Layer1
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # FC Layer2
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(fpconst.regions, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    train_error = model.evaluate(x_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Train loss:', train_error[0])
    print('Train accuracy:', train_error[1])
    return model


# ----------------------------------------------------------------------------------------------------------------------
# Tensorflow unsupervised model for vector generation

def convert_to_shape(arr):
    # print(max_height)
    height = arr.shape[0]

    total_pad = fpconst.max_height - height
    # print(total_pad)
    pad_height_up = int(total_pad / 2)
    pad_height_down = total_pad - pad_height_up

    img = np.zeros((fpconst.max_height, fpconst.max_width))
    start = pad_height_up
    end = pad_height_up + height
    # print(start)
    # print(end)

    img[start:end, :] = arr

    return img


def get_bounded_img(img_location):
    print(img_location)
    arr = misc.imread(img_location)
    minx = arr.shape[0]
    miny = arr.shape[1]
    maxx = 0
    maxy = 0

    # img = Image.fromarray(arr)
    # print(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i][j] != 255:
                minx = min(minx, i)
                maxx = max(maxx, i)
                miny = min(miny, j)
                maxy = max(maxy, j)

    arr = arr[minx:maxx, miny:maxy]

    img = Image.fromarray(arr)

    next_width = fpconst.max_width
    next_height = int(next_width * (img.height / img.width))
    img = img.resize((next_width, next_height), Image.ANTIALIAS)
    arr = np.array(img)
    arr = convert_to_shape(arr)
    return arr


# Returns train_data in the shape of m* (3*region_height * 3 * region_width * 1 )
def extract_from_image(file_name):
    img = get_bounded_img(file_name)
    height = fpconst.region_height_size
    width = fpconst.region_width_size

    img_rows = img.shape[0]
    img_cols = img.shape[1]

    # print(img_rows, img_cols)
    # print(height,width)

    result = []

    start_row = 0
    start_col = 0

    while True:
        # print("Current Row ::", start_row)
        # print("Current Col ::", start_col)

        if start_row >= img_rows:
            break
        data = np.zeros((3 * height, 3 * width))
        ## Extract Data

        row = start_row - height
        col = start_col - width
        count = 1
        r = 0
        c = 0
        while count < 9:
            # print(count)
            # print(r,r+height,c,c+width)
            # print(row,row+height,col,col+width)
            if row >= 0 and (row + height) < img_rows and col >= 0 and (col + width) < img_cols:
                data[r:r + height, c:c + width] = img[row:row + height, col:col + width]
            if count % 3 == 0:
                row = row + height
                col = start_col - width
                c = 0
                r = r + height
            else:
                col = col + width
                c = c + width

            count += 1

        data = data.reshape((3 * height, 3 * width, 1))
        result.append(data)

        # input("Test")

        start_col = start_col + width
        if start_col > img_cols:
            start_row = start_row + height
            start_col = 0

    return result


def get_data_image(loc):
    train_data = []
    for file in os.listdir(loc):
        file_name = loc + "/" + file
        data = extract_from_image(file_name)
        print("Image Extracted")
        for i in range(len(data)):
            train_data.append(data[i])
        # break  # TODO

    print(len(train_data))

    return np.array(train_data)


def initialize_gaussian_parameters_1D():
    tf.set_random_seed(1)  # so that your "random" numbers match ours

    x = tf.constant(np.random.uniform(-5, 5), name='x')

    mean_parameters = {
        0: tf.Variable(0.0, name="mu1", trainable=False),
        1: tf.Variable(0.0, name="mu2", trainable=False),
        2: tf.Variable(0.0, name="mu3", trainable=False),
        3: tf.Variable(0.0, name="mu4", trainable=False),
        4: tf.Variable(0.0, name="mu5", trainable=False),
        5: tf.Variable(0.0, name="mu6", trainable=False),
        6: tf.Variable(0.0, name="mu7", trainable=False),
        7: tf.Variable(0.0, name="mu8", trainable=False),
        8: tf.Variable(0.0, name="mu9", trainable=False)}

    sigma_parameters = {
        0: tf.Variable(0.0, name="sigma1", trainable=False),
        1: tf.Variable(0.0, name="sigma2", trainable=False),
        2: tf.Variable(0.0, name="sigma3", trainable=False),
        3: tf.Variable(0.0, name="sigma4", trainable=False),
        4: tf.Variable(0.0, name="sigma5", trainable=False),
        5: tf.Variable(0.0, name="sigma6", trainable=False),
        6: tf.Variable(0.0, name="sigma7", trainable=False),
        7: tf.Variable(0.0, name="sigma8", trainable=False),
        8: tf.Variable(0.0, name="sigma9", trainable=False)}

    pearson_parameter = {
        0: tf.Variable(np.random.uniform(-1, 1), name='P_1_0', dtype=tf.float32),
        1: tf.Variable(np.random.uniform(-1, 1), name='P_2_1', dtype=tf.float32),
        2: tf.Variable(np.random.uniform(-1, 1), name='P_3_2', dtype=tf.float32),
        3: tf.Variable(np.random.uniform(-1, 1), name='P_4_3', dtype=tf.float32),
        4: tf.Variable(np.random.uniform(-1, 1), name='P_5_4', dtype=tf.float32),
        5: tf.Variable(np.random.uniform(-1, 1), name='P_6_5', dtype=tf.float32),
        6: tf.Variable(np.random.uniform(-1, 1), name='P_7_6', dtype=tf.float32),
        7: tf.Variable(np.random.uniform(-1, 1), name='P_8_7', dtype=tf.float32)
    }
    return mean_parameters, sigma_parameters, pearson_parameter


def initialize_gaussian_parameters():
    parameters = {}
    tf.set_random_seed(1)  # so that your "random" numbers match ours

    mean_parameters = {0: tf.get_variable('mu_0', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       1: tf.get_variable('mu_1', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       2: tf.get_variable('mu_2', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       3: tf.get_variable('mu_3', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       4: tf.get_variable('mu_4', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       5: tf.get_variable('mu_5', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       6: tf.get_variable('mu_6', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       7: tf.get_variable('mu_7', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                       8: tf.get_variable('mu_8', [dimensions, 1],
                                          initializer=tf.contrib.layers.xavier_initializer(seed=0))}

    parameters["mean"] = mean_parameters

    sigma_parameters = {"01": tf.get_variable('S_0_1', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "10": tf.get_variable('S_1_0', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "12": tf.get_variable('S_1_2', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "21": tf.get_variable('S_2_1', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "23": tf.get_variable('S_2_3', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "32": tf.get_variable('S_3_2', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "34": tf.get_variable('S_3_4', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "43": tf.get_variable('S_4_3', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "45": tf.get_variable('S_4_5', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "54": tf.get_variable('S_5_4', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "56": tf.get_variable('S_5_6', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "65": tf.get_variable('S_6_5', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "67": tf.get_variable('S_6_7', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "76": tf.get_variable('S_7_6', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "78": tf.get_variable('S_7_8', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "87": tf.get_variable('S_8_7', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "00": tf.get_variable('S_0_0', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "11": tf.get_variable('S_1_1', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "22": tf.get_variable('S_2_2', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "33": tf.get_variable('S_3_3', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "44": tf.get_variable('S_4_4', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "55": tf.get_variable('S_5_5', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "66": tf.get_variable('S_6_6', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "77": tf.get_variable('S_7_7', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0)),
                        "88": tf.get_variable('S_8_8', [dimensions, dimensions],
                                              initializer=tf.contrib.layers.xavier_initializer(seed=0))}

    parameters["sigma"] = sigma_parameters
    return parameters


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    final_parameters = {}

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    W1 = tf.get_variable('W_0_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_0_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters1 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_0"] = parameters1

    W1 = tf.get_variable('W_1_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_1_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters2 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_1"] = parameters2

    W1 = tf.get_variable('W_2_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_2_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters3 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_2"] = parameters3

    W1 = tf.get_variable('W_3_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_3_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters4 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_3"] = parameters4

    W1 = tf.get_variable('W_4_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_4_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters5 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_4"] = parameters5

    W1 = tf.get_variable('W_5_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_5_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters6 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_5"] = parameters6

    W1 = tf.get_variable('W_6_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_6_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters7 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_6"] = parameters7

    W1 = tf.get_variable('W_7_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_7_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters8 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_7"] = parameters8

    W1 = tf.get_variable('W_8_1', [4, 4, 1, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W_8_2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters9 = {"W1": W1,
                   "W2": W2}

    final_parameters["W_8"] = parameters9

    return final_parameters


def random_mini_batches(X, mini_batch_size=64, seed=0):
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

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_mini_batches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_mini_batches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch = mini_batch_X
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_mini_batches * mini_batch_size:, :]
        mini_batch = mini_batch_X
        mini_batches.append(mini_batch)

    return mini_batches


def get_conditional_distribution(mean, sigma, variable_index_1, variable_index_2, value_2):
    mu_1 = mean[variable_index_1]
    mu_2 = mean[variable_index_2]
    sigma_11 = sigma[(str(variable_index_1) + str(variable_index_1))]
    sigma_12 = sigma[(str(variable_index_1) + str(variable_index_2))]
    sigma_21 = sigma[(str(variable_index_2) + str(variable_index_1))]
    sigma_22 = sigma[(str(variable_index_2) + str(variable_index_2))]

    sigma_22_inverse = tf.linalg.inv(sigma_22)
    multiplier = tf.matmul(sigma_12, sigma_22_inverse)
    mu_new = tf.add(mu_1, tf.matmul(multiplier, value_2 - mu_2))
    sigma_new = tf.subtract(sigma_11, tf.matmul(multiplier, sigma_21))

    return tf.contrib.distributions.Normal(loc=mu_new, scale=sigma_new)


def compute_cost(Z, parameters):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    z_0, mu_0, sigma_0, a, b = Z[0]
    z_1, mu_1, sigma_1, mu_1_0, sigma_1_0 = Z[1]
    z_2, mu_2, sigma_2, mu_2_1, sigma_2_1 = Z[2]
    z_3, mu_3, sigma_3, mu_3_2, sigma_3_2 = Z[3]
    z_4, mu_4, sigma_4, mu_4_3, sigma_4_3 = Z[4]
    z_5, mu_5, sigma_5, mu_5_4, sigma_5_4 = Z[5]
    z_6, mu_6, sigma_6, mu_6_5, sigma_6_5 = Z[6]
    z_7, mu_7, sigma_7, mu_7_6, sigma_7_6 = Z[7]
    z_8, mu_8, sigma_8, mu_8_7, sigma_8_7 = Z[8]

    # P(z_0 ,z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8)
    # approx P(z_0)P(z_1|z_0)P(z_2|z_1)P(z_3|z_2)P(z_4|z_3)P(z_5|z_4)P(z_6|z_5)P(z_7|z_6)P(z_8|z_7)

    gaussian_dist_0 = tf.contrib.distributions.Normal(loc=mu_0, scale=sigma_0)
    gaussian_dist_1_0 = tf.contrib.distributions.Normal(loc=mu_1_0, scale=sigma_1_0)
    gaussian_dist_2_1 = tf.contrib.distributions.Normal(loc=mu_2_1, scale=sigma_2_1)
    gaussian_dist_3_2 = tf.contrib.distributions.Normal(loc=mu_3_2, scale=sigma_3_2)
    gaussian_dist_4_3 = tf.contrib.distributions.Normal(loc=mu_4_3, scale=sigma_4_3)
    gaussian_dist_5_4 = tf.contrib.distributions.Normal(loc=mu_5_4, scale=sigma_5_4)
    gaussian_dist_6_5 = tf.contrib.distributions.Normal(loc=mu_6_5, scale=sigma_6_5)
    gaussian_dist_7_6 = tf.contrib.distributions.Normal(loc=mu_7_6, scale=sigma_7_6)
    gaussian_dist_8_7 = tf.contrib.distributions.Normal(loc=mu_8_7, scale=sigma_8_7)

    print("Shape", z_0.get_shape())
    # print("Shape", z_8.get_shape())
    # print("Shape", mean[0].get_shape())
    # print("Shape", sigma["00"].get_shape())
    #
    log_prob = gaussian_dist_0.log_prob(z_0)

    log_prob += gaussian_dist_1_0.log_prob(z_1)
    log_prob += gaussian_dist_2_1.log_prob(z_2)
    log_prob += gaussian_dist_3_2.log_prob(z_3)
    log_prob += gaussian_dist_4_3.log_prob(z_4)
    log_prob += gaussian_dist_5_4.log_prob(z_5)
    log_prob += gaussian_dist_6_5.log_prob(z_6)
    log_prob += gaussian_dist_7_6.log_prob(z_7)
    log_prob += gaussian_dist_8_7.log_prob(z_8)

    print("Prob Shape", log_prob.get_shape())

    neg_log_likelihood = -1.0 * tf.reduce_mean(log_prob)

    return neg_log_likelihood


def forward_propagation(X, parameters, mean, sigma, prev_value=tf.constant(0.0), pearson=tf.constant(1.0), prev_mean=tf.constant(1.0),
                        prev_sigma=tf.constant(1.0)):
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    # print(P2.get_shape())
    P2 = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2, dimensions, activation_fn=None)

    Z4 = tf.contrib.layers.fully_connected(Z3, 1, activation_fn=None)
    # Z3 = tf.expand_dims(Z3,2)
    val = tf.nn.moments(Z4, axes=0)
    mu = val[0]
    s = val[1]
    mean = alpha * mean + (1 - alpha) * mu
    sigma = beta * sigma + (1 - beta) * s

    sigma = sigma + tf.constant(0.001)

    pearson = tf.sigmoid(pearson)
    pearson = tf.multiply(pearson, 2)
    pearson = tf.subtract(pearson, tf.constant(1.0))

    # pearson = tf.Print(pearson,[pearson],"Pearson Coefficient")

    prev_value = tf.reduce_mean(prev_value)

    factor = tf.divide(sigma, prev_sigma)
    factor = tf.multiply(factor, tf.subtract(prev_value, prev_mean))
    factor = tf.multiply(pearson, factor)
    mu_new = tf.add(mean, factor)

    # sigma = sigma_1(1 - pearson^2)
    sigma_new = tf.multiply(sigma, tf.subtract(tf.constant(1.0), tf.pow(pearson, tf.constant(2.0))))

    mu_new = tf.multiply(mu_new, tf.constant(1.0))
    sigma_new = tf.multiply(sigma_new,tf.constant(1.0))

    return Z4, mean, sigma, mu_new, sigma_new


# train_data is of shape (3*region_height * 3*region_width * 1 * m)
def get_model_tf(train_data):
    seed = 10
    print_cost = True
    m = train_data.shape[2]

    n_h = fpconst.region_height_size
    n_w = fpconst.region_width_size
    n_c = 1

    tf_regions = []

    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_0"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_1"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_2"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_3"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_4"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_5"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_6"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_7"))
    tf_regions.append(tf.placeholder(tf.float32, shape=[None, n_h, n_w, 1], name="x_region_8"))

    parameters = initialize_parameters()

    mean_parameters, sigma_parameters, gaussian_parameters = initialize_gaussian_parameters_1D()

    Z = []
    for i in range(9):
        Z.append(None)
        tf_region = tf_regions[i]
        parameter_key = "W_" + str(i)
        if i == 0:
            Z[i] = forward_propagation(tf_region, parameters[parameter_key], mean_parameters[i], sigma_parameters[i])
        else:
            Z[i] = forward_propagation(tf_region, parameters[parameter_key], mean_parameters[i], sigma_parameters[i],
                                       Z[i - 1][0], gaussian_parameters[i - 1], Z[i - 1][1], Z[i - 1][2])

        mean_parameters[i] = Z[i][0]
        sigma_parameters[i] = Z[i][1]

    cost = compute_cost(Z, gaussian_parameters)

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
                mini_batches = random_mini_batches(train_data, mini_batch_size, seed)

                for mini_batch in mini_batches:
                    if mini_batch.shape[0] == 0:
                        continue
                    # Select a mini_batch
                    x_train_region_0 = mini_batch[:, 0:n_h, 0:n_w, :]
                    x_train_region_1 = mini_batch[:, 0:n_h, n_w:2 * n_w, :]
                    x_train_region_2 = mini_batch[:, 0:n_h, 2 * n_w:3 * n_w, :]

                    x_train_region_3 = mini_batch[:, n_h:2 * n_h, 0:n_w, :]
                    x_train_region_4 = mini_batch[:, n_h:2 * n_h, n_w:2 * n_w, :]
                    x_train_region_5 = mini_batch[:, n_h:2 * n_h, 2 * n_w:3 * n_w, :]

                    x_train_region_6 = mini_batch[:, 2 * n_h:3 * n_h, 0:n_w, :]
                    x_train_region_7 = mini_batch[:, 2 * n_h:3 * n_h, n_w:2 * n_w, :]
                    x_train_region_8 = mini_batch[:, 2 * n_h:3 * n_h, 2 * n_w:3 * n_w, :]

                    # IMPORTANT: The line that runs the graph on a mini_batch.
                    _, temp_cost = sess.run([optimizer, cost],
                                            feed_dict={tf_regions[8]: x_train_region_8, tf_regions[0]: x_train_region_0,
                                                       tf_regions[1]: x_train_region_1, tf_regions[2]: x_train_region_2,
                                                       tf_regions[3]: x_train_region_3, tf_regions[4]: x_train_region_4,
                                                       tf_regions[5]: x_train_region_5, tf_regions[6]: x_train_region_6,
                                                       tf_regions[7]: x_train_region_7})

                    mini_batch_cost += temp_cost / num_mini_batches

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

    return parameters


def train():
    location = fpconst.locations[0]
    X_train = get_data_image(location)
    get_model_tf(X_train)


if __name__ == "__main__":
    train()
    # location = fpconst.save_locations[0]
    # x_train, y_train, x_test, y_test, input_shape = get_data(location)
    # model = get_model(x_train, y_train, x_test, y_test, input_shape)
