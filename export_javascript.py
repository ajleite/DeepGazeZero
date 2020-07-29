#!/usr/bin/python3

# Dependencies: tensorflow 2.x
#               numpy
#               tensorflowjs
# To install tensorflow x.0 you may first need to update your pip.
# Be wary of installing tensorflow 2.x using pip and Anaconda;
# it may be simpler to install in a separate Python installation.

# This file invokes Deep Gaze Zero model in a javascript wrapper.

import sys, os, json

import numpy as np
import tensorflow as tf
import tensorflowjs

from model import CNN, parameter_dimension, apply_model

class VentralStreamModel(tf.Module):
    def __init__(self):
        self.CNN = CNN
    @tf.function(input_signature=[tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32), tf.TensorSpec(shape=(parameter_dimension,), dtype=tf.float32)])
    def __call__(self, image, parameters):
        return apply_model(image, parameters, js_safe=True)

if __name__ == '__main__':
    vsmodel = VentralStreamModel()
    tf.saved_model.save(vsmodel, "vs_model.savedmodel")
    tensorflowjs.converters.convert_tf_saved_model('vs_model.savedmodel', 'vs_model.tfjs')

    params = np.load('params.npy')
    json.dump(params.tolist(), open('vs_model.tfjs/params.json', 'w'))
