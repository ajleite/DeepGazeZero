#!/usr/bin/python3

# Dependencies: tensorflow 2.x
#               numpy
# To install tensorflow 2.x you may first need to update your pip.
# Be wary of installing tensorflow 2.x using pip and Anaconda;
# it may be simpler to install in a separate Python installation.

# This file takes the Judd dataset to working Python data.
# First, fixation data must be extracted from MATLAB to JSON-compatible array
# format using Judd's code and my modified "showEyeDataAcrossUsers.m" (included).

import os
import random
import pickle

import tensorflow as tf
import numpy as np

def load_and_resize_image_and_fixations(image_path, fixations_path):
    unresized_image = tf.image.decode_image(open(image_path,'rb').read())
    resized_image = tf.image.resize(unresized_image,(224,224), antialias=True)/255

    image_dims = unresized_image.shape

    points = np.array(eval(open(fixations_path).read()))
    points = np.flip(points, axis=1) # switch x, y
    points[:,0] *= 224
    points[:,0] //= image_dims[0]
    points[:,1] *= 224
    points[:,1] //= image_dims[1]

    # Judd also discards out-of-bounds data.
    restricted_points = points[(0 <= points[:,0]) & (points[:,0] < 224) & (0 <= points[:,1]) & (points[:,1] < 224)]

    return resized_image.numpy(), restricted_points

def import_dataset(image_folder, fixations_folder):
    tuples = {}
    for fn in os.listdir(image_folder):
        if not fn.endswith('.jpg') or fn.endswith('.jpeg'):
            continue
        image_path = os.path.join(image_folder, fn)
        fixations_path = os.path.join(fixations_folder, fn+'.json')
        if not os.path.isfile(fixations_path):
            print(f'Warning: can\'t find fixations for {fn}... Skipping.')
            continue
        image, fixations = load_and_resize_image_and_fixations(image_path, fixations_path)
        tuples[fn] = (image, fixations)
    return tuples

def shuffle_and_split_dataset(dataset, train_fraction=0.5, seed=42):
    rand = random.Random(seed)
    dataset_list = [(fn, image, fixations) for fn, (image, fixations) in dataset.items()]
    rand.shuffle(dataset_list)
    train_data_list = dataset_list[:int(len(dataset_list)*train_fraction)]
    train_data = {fn: (image, fixations) for fn, image, fixations in train_data_list}
    test_data_list = dataset_list[int(len(dataset_list)*train_fraction):]
    test_data = {fn: (image, fixations) for fn, image, fixations in test_data_list}
    return train_data, test_data

def store_dataset(train_data, test_data, filename):
    file = open(filename, 'wb')
    pickle.dump((train_data, test_data), file)
    file.close()

def load_dataset(filename):
    file = open(filename, 'rb')
    train_data, test_data = pickle.load(file)
    file.close()
    return train_data, test_data

if __name__ == '__main__':
    # each jpg image in images should correspond to a json file containing fixations in fixations.
    train_data, test_data = import_dataset('images', 'fixations')
    store_dataset(train_data, test_data, 'Judd_dataset.pickle')
