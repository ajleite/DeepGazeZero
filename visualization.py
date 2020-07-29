#!/usr/bin/python3

# Dependencies: tensorflow 2.x
#               numpy
#               matplotlib.pyplot
# To install tensorflow 2.x you may first need to update your pip.
# Be wary of installing tensorflow 2.x using pip and Anaconda;
# it may be simpler to install in a separate Python installation.

# This file contains code for visualizing the NSS calculation.

import sys, os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import apply_model

def display_NSS_calculation(image, fixations, parameters, figure=None, title=None):
    prediction = apply_model(image, parameters)

    plt.figure(1, (15,6))
    plt.subplot(1,3,1)
    plt.title('(a) Stimulus')
    plt.imshow(image)

    # want hunk_size sized bins; want to cover full data range
    normalized = (prediction-tf.reduce_mean(prediction)) / tf.math.reduce_std(prediction)
    NSS_values = [normalized[x, y] for x, y in fixations]
    NSS = np.mean(NSS_values)

    hunk_size = 1
    smallest = np.min(normalized)//hunk_size * hunk_size
    biggest = (np.max(normalized)//hunk_size + 2) * hunk_size
    hunks = np.arange(smallest,biggest,hunk_size)

    axes = plt.subplot(1,3,2)
    plt.title('(b) Predicted salience map (normalized)\n and actual fixation points')
    plt.imshow(normalized, cmap='Greys_r', vmin=-np.max(np.abs(normalized)), vmax=np.max(np.abs(normalized)))
    contour = plt.contour(np.flip(normalized, axis=0), hunks, colors='k', linewidths=1, origin='image')
    if len(fixations):
        plt.scatter(fixations[:,1], fixations[:,0], marker='.')
    axes.clabel(contour, fmt='%1d', colors='#006600', inline=1, fontsize=9)

    axes = plt.subplot(1,3,3)
    plt.title(f'(c) Calculating normalized scanpath saliency:\nhistogram and mean (NSS = {NSS:.2f})')
    plt.hist(NSS_values, bins=hunks)
    axes.axvline(x=NSS, color='black', linestyle='--')
    axes.set_xticks(hunks)
    axes.set_xticklabels([str(i) for i in hunks])
    axes.set_aspect(1.0/axes.get_data_ratio()) # unfortunately needed for square subplot

    if title is not None:
        plt.suptitle(f'{title}\n(NSS: {NSS:.4f})')

    if figure is not None:
        plt.savefig(figure, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return NSS