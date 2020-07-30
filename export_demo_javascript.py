#!/usr/bin/python3

# Dependencies: tensorflow 2.x
#               numpy
#               tensorflowjs
# To install tensorflow x.0 you may first need to update your pip.
# Be wary of installing tensorflow 2.x using pip and Anaconda;
# it may be simpler to install in a separate Python installation.

# This file invokes Deep Gaze Zero model in a javascript wrapper,
# separating out the different levels of features for a more
# instructive approach.

import sys, os, json

import numpy as np
import tensorflow as tf
import tensorflowjs

from model import CNN, layers, parameter_dimension, filter_edges

@tf.function
def apply_model(image, parameters=None, standardize=False, js_safe=False):
    # ignore alpha channel
    if image.shape[2] == 4:
        image = image[:,:,:3]

    # Start the tally of total weighted feature activation using the color features.
    image_dims = image.shape[0:2]
    if parameters is not None:
        # xyf,f->xy
        image_level = tf.tensordot(image, parameters[:3], axes=[[2],[0]])
    else:
        image_level = tf.reduce_mean(image, axis=2)
    params_index = 3

    # Put the image through the CNN, making use of any ReLU units as feature maps.
    feature_levels = {224: image_level}
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

            size = prioritized.shape[0]
            if not size in feature_levels:
                feature_levels[size] = scaled_up
            else:
                feature_levels[size] += scaled_up

    LL = feature_levels[224] + feature_levels[112] + feature_levels[56]
    ML = feature_levels[28] + feature_levels[14]
    HL = feature_levels[7]
    total = LL + ML + HL
    return [total, LL, ML, HL]

class VentralStreamDemoModel(tf.Module):
    def __init__(self):
        self.CNN = CNN
    @tf.function(input_signature=[tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32), tf.TensorSpec(shape=(parameter_dimension,), dtype=tf.float32)])
    def __call__(self, image, parameters):
        return apply_model(image, parameters, js_safe=True)

if __name__ == '__main__':
    vsdemomodel = VentralStreamDemoModel()
    call_output = vsdemomodel.__call__.get_concrete_function(tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32), tf.TensorSpec(shape=(parameter_dimension,), dtype=tf.float32))
    tf.saved_model.save(vsdemomodel, "vs_model_demo.savedmodel", signatures={'serving_default': call_output})
    tensorflowjs.converters.convert_tf_saved_model('vs_model_demo.savedmodel', 'vs_model_demo.tfjs')

    params = np.load('params.npy')
    json.dump(params.tolist(), open('vs_model_demo.tfjs/params.json', 'w'))
