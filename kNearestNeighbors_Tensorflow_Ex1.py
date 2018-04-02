#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:54:24 2018
@author: dilekcelik

This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf

#import data set
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#Limiting Mnist Data
X_train, Y_train = mnist.train.next_batch(5000) #5000 for training (nn candidates)
X_test, Y_test = mnist.test.next_batch(200) #200 for test


#Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(X_train, tf.negative(X_train))), reduction_indices=1)
#prediction: Get in distance index (Nearest neighbor)
pred = tf.arg_min(distance,0)

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # loop over test data
    for i in range(len(X_test)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: X_train, xte: X_test[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Y_train[nn_index]), \
            "True Class:", np.argmax(Y_test[i]))
        # Calculate accuracy
        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    print("Done!")
    print("Accuracy:", accuracy)


