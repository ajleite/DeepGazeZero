# Deep Gaze Zero
### Abe Leite, Indiana University Bloomington

Deep Gaze Zero is a model of visual salience computation in the human brain with significant algorithmic similarities to Matthias Kümmerer's Deep Gaze I and II models.

In Deep Gaze Zero, visual salience is represented as the weighted average of feature activation maps, which are calculated by a neural network previously trained on image recognition.

The actual Deep Gaze Zero model, found in `model.py`, is extremely simple, tallying 54 lines of code excluding whitespace and comments. Also included are tools to interface with Tilke Judd's 2009 dataset (`importJuddData.m`, `judd_dataset.py`), export the model for use in tensorflow.js (`export_javascript.py`), train and test the model (`train_test.py`), and visualize the model's processing (`visualization.py`).

### Dependencies

This code runs on Python 3.7+, with additional dependencies on tensorflow 2.x, numpy, pyplot (optional), and tensorflowjs (optional). To install all of these to your site-packages, run

    python3 -m pip install tensorflow matplotlib numpy tensorflowjs

If you are in a virtual environment, you may have an alternative way of invoking pip.

### Included Trained Parameters

I have included `params.npy`, which is a feature weighting I fitted to half of Judd's dataset for 10 epochs, and `params_standardize.npy`, a variant feature weighting optimized for standardized feature activity maps, which may be enabled by modifying `model.py`. (This is the approach taken in Deep Gaze I.)

### Web Demo

To see an illustrative demo of the model on an image of your choice or on your webcam data, visit [https://ajleite.github.io/DeepGazeZero].
