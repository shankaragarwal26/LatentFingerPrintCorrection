from __future__ import print_function

import os
import sys
from math import ceil, floor

import numpy as np

#
from PIL import Image

sys.path.append("/Users/shankaragarwal/LP/LatentFingerPrintCorrection/fingerprint_python")
# sys.path.append("/home/ubuntu/code/fingerprint_python")

from constants import constants as fpconst
from preprocessor import get_images as get_images
from encoder_cnn import encoder
import tensorflow as tf

IMAGE_HEIGHT = fpconst.region_height_size
IMAGE_WIDTH = fpconst.region_width_size


def central_scale_images(X_imgs, scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.int32)

    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)

    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    return X_scale_data


def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_HEIGHT, ceil(0.8 * IMAGE_WIDTH)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_WIDTH))
        h_start = 0
        h_end = IMAGE_HEIGHT
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_HEIGHT, ceil(0.8 * IMAGE_WIDTH)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_WIDTH))
        w_end = IMAGE_WIDTH
        h_start = 0
        h_end = IMAGE_HEIGHT
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_HEIGHT), IMAGE_WIDTH], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_WIDTH
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_HEIGHT))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_HEIGHT), IMAGE_WIDTH], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_WIDTH
        h_start = int(floor((1 - 0.8) * IMAGE_HEIGHT))
        h_end = IMAGE_HEIGHT

    return offset, size, w_start, w_end, h_start, h_end


def translate_images(X_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_HEIGHT, IMAGE_WIDTH, 1),
                                    dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
            w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr


def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * np.pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={X: img})
            X_flip.extend(flipped_imgs)

    X_flip = np.array(X_flip)
    return X_flip


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, max(i - 1, 1), int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, max(i - 1, 1), int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype=np.int32)
    return vertices


def remove_parts_of_img(X_imgs, probs):
    images = []
    for prob in probs:
        for i in range(X_imgs.shape[0]):
            img = X_imgs[i]
            res = img.copy()
            res = res.reshape(img.shape[0], img.shape[1])
            rows, cols = res.shape
            for row in range(rows):
                for col in range(cols):
                    rand = np.random.rand()
                    if rand <= prob:
                        res[row, col] = 255
            res = res.reshape((rows, cols, 1))
            images.append(res)
    return images


def save_images(count, folder_name, images, prefix):
    n = images.shape[0]
    for j in range(n):
        tr_img = images[j]
        noise_file_name = folder_name + "/" + prefix + str(count)
        np.save(noise_file_name, tr_img)
        count = count + 1
    return count


def save_data(list, prefixes, y, height, width, folder_name):
    count = 1
    y = y.reshape((1, height, width, 1))
    prefix = "noise"
    images = y
    val = False
    if list[0] == 1:
        images = translate_images(images)
        prefix = prefix + "_" + prefixes[0]
        val = True
    if list[1] == 1:
        images = rotate_images(images, -180, 180, 25)
        prefix = prefix + "_" + prefixes[1]
        val = True
    if list[2] == 1:
        images = np.array(remove_parts_of_img(images, [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8]))
        prefix = prefix + "_" + prefixes[2]
        val = True
    if list[3] == 1:
        images = central_scale_images(images, [0.75, 0.60, 0.50, 0.25])
        prefix = prefix + "_" + prefixes[3]
        val = True
    if list[4] == 1:
        images = flip_images(images)
        prefix = prefix + "_" + prefixes[4]
        val = True
    if list[5] == 1:
        images = add_salt_pepper_noise(images)
        prefix = prefix + "_" + prefixes[5]
        val = True
    if val:
        return save_images(count, folder_name, images, prefix)


def save_recursive(list, prefixes, index, y, height, width, folder_name):
    if index == 6:
        return
    list[index] = 1
    save_data(list, prefixes, y, height, width, folder_name)
    save_recursive(list, prefixes, index + 1, y, height, width, folder_name)
    list[index] = 0
    save_recursive(list, prefixes, index + 1, y, height, width, folder_name)


def train(loc, encoder_model):
    out_loc = fpconst.output
    Y = get_images.extract_all_regions(loc)
    m = Y.shape[0]
    for i in range(m):
        folder_name = out_loc + '/' + str(i)
        try:
            os.makedirs(folder_name)
        except Exception as e:
            pass
        folder_name = folder_name + "/"
        orig_file_name = folder_name + fpconst.ORIGINAL_FILE_NAME
        y = Y[i]
        np.save(orig_file_name, y)
        list = [0, 0, 0, 0, 0, 0]
        prefixes = ["tr", "ro", "bl", "sc", "fl", "sap"]
        save_recursive(list, prefixes, 0, y, Y.shape[1], Y.shape[2], folder_name)


train(fpconst.locations[0], None)
