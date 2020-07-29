#!/usr/bin/python3

# Dependencies: tensorflow 2.x
#               numpy
# To install tensorflow x.0 you may first need to update your pip.
# Be wary of installing tensorflow 2.x using pip and Anaconda;
# it may be simpler to install in a separate Python installation.

# This file contains the Deep Gaze Zero model, inspired by Deep Gaze I by Matthias Kümmerer.

# The Deep Gaze Zero model differs from Deep Gaze I in two respects,
# both in the pursuit of biological plausibility:
# - First, feature maps are not standardized before they are input into the model.
#   The direct feature activation maps from the neural network's units are used.
# - Second, the output map is not normalized in any way. This means that the values
#   of salience maps are not predictable, but crucially it means that two images'
#   salience maps can be compared to see which image was more stimulating.

import sys, os

import numpy as np
import tensorflow as tf

# This works for any strictly feed-forward CNN.
CNN = tf.keras.applications.MobileNet(weights='imagenet')
layers = CNN.layers
parameter_dimension = 3 + sum(layer.output_shape[-1] for layer in layers if isinstance(layer, tf.keras.layers.ReLU))
print(f'Expecting {parameter_dimension}-parameter models.')

@tf.function
def filter_edges(output):
    ''' MobileNet exhibits artifacts on the edges of the image. This code sets the
        edges to the minimum values for each filter -- a good way of getting around
        some of the artifacting exhibited. This introduces a small center bias. '''
    height, width, channel = output.shape
    middle = output[1:-1,1:-1,:]
    smallest_value = tf.reduce_min(middle, axis=[0,1], keepdims=True)
    horiz_zeros = tf.zeros((1, width-2, channel), output.dtype) + smallest_value
    topbottom_filtered = tf.concat([horiz_zeros, middle, horiz_zeros], axis=0)
    vert_zeros = tf.zeros((height, 1, channel), output.dtype) + smallest_value
    filtered = tf.concat([vert_zeros, topbottom_filtered, vert_zeros], axis=1)
    return filtered

@tf.function
def apply_model(image, parameters=None, standardize=False, js_safe=False):
    # ignore alpha channel
    if image.shape[2] == 4:
        image = image[:,:,:3]

    # Start the tally of total weighted feature activation using the color features.
    image_dims = image.shape[0:2]
    if parameters is not None:
        # xyf,f->xy
        total = tf.tensordot(image, parameters[:3], axes=[[2],[0]])
    else:
        total = tf.reduce_mean(image, axis=2)
    params_index = 3

    # Put the image through the CNN, making use of any ReLU units as feature maps.
    output = image
    for layer in layers:
        # Should just be output = layer(output), but keras assumes you have batch dim,
        # thus the expand_dims and squeeze.
        output = tf.expand_dims(output, axis=0)
        output = layer(output)
        output = tf.squeeze(output, axis=0)

        # Make use of activated values only
        if isinstance(layer, tf.keras.layers.ReLU):
            # Filter out edge artifacting.
            filtered = filter_edges(output)
            # Optionally standardize as in Deep Gaze I
            if standardize:
                filtered -= tf.reduce_mean(filtered, axis=[0,1], keepdims=True)
                stds = tf.math.reduce_std(filtered, axis=[0,1], keepdims=True)
                multiplier = tf.where(stds > 0, 1/stds, 0)
                filtered *= multiplier

            # In the paper, the linear combination is written as occurring over all
            # feature maps simultaneously after scaling up.
            # However, I've found this individual approach to be more memory-efficient
            # for both training and inference.
            if parameters is not None:
                num_features = int(filtered.shape[-1])
                next_index = params_index + num_features
                # xyf,f->xy
                prioritized = tf.tensordot(filtered, parameters[params_index:next_index], axes=[[2],[0]])
                params_index = next_index
            else:
                prioritized = tf.reduce_mean(image, axis=2)

            # expand_dims and squeeze on the channel axis because tf.image.resize
            # expects width x height x channel
            prioritized = tf.expand_dims(prioritized, axis=-1)
            # Optionally use a worse scaling method (for the javascript version)
            if js_safe:
                scaled_up = tf.image.resize(prioritized, image_dims)
            else:
                scaled_up = tf.image.resize(prioritized, image_dims, method='gaussian')
            scaled_up = tf.squeeze(scaled_up, axis=-1)

            total += scaled_up
    return total

if __name__ == '__main__':
    parameters = tf.Variable(tf.ones((parameter_dimension,))/parameter_dimension)
    parameters.assign(np.load('params.npy'))

    if len(sys.argv) < 2:
        print('Usage: model.py [image1] [image2] ...')
    for filename in sys.argv[1:]:
        unresized_image = tf.image.decode_image(open(filename,'rb').read())
        image = tf.image.resize(unresized_image,(224,224), antialias=True)/255
        prediction = apply_model(image, parameters).numpy()
        safe_filename = filename.replace('/', '_')
        np.save('out/'+safe_filename+'.npy', prediction)
        try:
            import matplotlib.pyplot as plt
            prediction_image = prediction - np.min(prediction)
            prediction_image /= np.max(prediction_image)
            prediction_image *= 255
            plt.imsave('out/'+safe_filename+'.png', prediction_image, cmap='Greys_r')
        except ImportError:
            print('Cannot save png as pyplot is not present.')
