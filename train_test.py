#!/usr/bin/python3

# Dependencies: tensorflow 2.x
#               numpy
# To install tensorflow x.0 you may first need to update your pip.
# Be wary of installing tensorflow 2.x using pip and Anaconda;
# it may be simpler to install in a separate Python installation.

# This file contains code training and testing a Deep Gaze Zero model
# to maximize its NSS -- the values assigned to actual fixation points
# in a standardized salience map.

import os
import sys
import random
import pickle

import tensorflow as tf
import numpy as np

from model import parameter_dimension, apply_model
from judd_dataset import load_dataset, shuffle_and_split_dataset
from visualization import display_NSS_calculation

@tf.function(input_signature=[tf.TensorSpec(shape=(224, 224), dtype=tf.float32), tf.TensorSpec(shape=(None, 2), dtype=tf.int64)])
def calculate_NSS(prediction, fixations):
    standardized = (prediction - tf.math.reduce_mean(prediction)) / tf.math.reduce_std(prediction)
    return tf.reduce_mean(tf.gather_nd(standardized, fixations))

@tf.function(input_signature=[tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, 2), dtype=tf.int64), tf.TensorSpec(shape=(parameter_dimension,), dtype=tf.float32)])
def parameter_gradient(image_sample, fixations, parameters):
    prediction = apply_model(image_sample, parameters)
    NSS = calculate_NSS(prediction, fixations)
    # gradients will return a list with a single element: [d(-NSS)/d(parameters)]
    grad = tf.gradients(-NSS, [parameters])[0]
    return NSS, grad

def train_epoch(train_data, parameters, optimizer, minibatch_size=12):
    NSS_scores = []
    minibatch = []
    for fn, (image, fixations) in train_data.items():
        NSS, grad = parameter_gradient(image, fixations, parameters)
        print(fn, NSS.numpy())
        NSS_scores.append(NSS.numpy())
        minibatch.append(grad)
        if len(minibatch) == minibatch_size:
            gradient = tf.reduce_mean(minibatch,axis=0)
            optimizer.apply_gradients([(gradient, parameters)])
            minibatch = []
    if minibatch:
        gradient = tf.reduce_mean(minibatch,axis=0)
        optimizer.apply_gradients([(gradient, parameters)])
    return np.mean(NSS_scores)

def test(test_data, parameters):
    NSS_scores = []
    for fn, (image, fixations) in test_data.items():
        prediction = apply_model(image, parameters)
        NSS = calculate_NSS(prediction, fixations)
        NSS_scores.append(NSS)
    return np.mean(NSS_scores)

def train(train_data, parameters, training_rate=2e-6, epochs=10, minibatch_size=12, out='params'):
    optimizer = tf.keras.optimizers.Adam(training_rate)
    actual_train_data, validate_data = shuffle_and_split_dataset(train_data, train_fraction=0.9, seed=80)
    for epoch in range(epochs):
        NSS = train_epoch(actual_train_data, parameters, optimizer)
        validate_NSS = test(validate_data, parameters)
        print(f'Epoch {epoch} NSS {NSS}\nValidation NSS {validate_NSS}\n\n\n\n')
        np.save(f'{out}.{epoch}.npy', parameters.numpy())

if __name__ == '__main__':
    parameters = tf.Variable(tf.ones((parameter_dimension,))/parameter_dimension)
    train_data, test_data = load_dataset('Judd_dataset.pickle')

    training = False
    testing = False

    if training:
        train(train_data, parameters)
        np.save('params.npy', parameters.numpy())
    else:
        parameters.assign(np.load('params.npy'))

    if testing:
        print('Test NSS:', test(test_data, parameters))

    NSS_scores = []

    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            unresized_image = tf.image.decode_image(open(filename,'rb').read())
            image = tf.image.resize(unresized_image,(224,224), antialias=True)/255
            prediction = apply_model(image, parameters)
            safe_filename = filename.replace('/', '_')
            np.save('out/'+safe_filename+'.npy', prediction.numpy())
            display_NSS_calculation(image, [], parameters, figure='out/'+safe_filename+'.pdf', title=filename)
    else:
        all_NSS = []
        for filename in test_data.keys():
            image, fixations = test_data[filename]

            figure_out = 'out/'+filename+'.pdf'
            NSS = display_NSS_calculation(image, fixations, parameters, figure=figure_out, title=None)
            print(filename, NSS)
            all_NSS.append(NSS)
        print('Mean: ', np.mean(all_NSS))
