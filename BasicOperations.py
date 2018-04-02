#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:19:35 2018

@author: dilekcelik
"""

from __future__ import print_function
import tensorflow as tf

# Basic constant operations
a=tf.constant(2)
b=tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))
    
# Basic Operations with variable as graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
# Define some operations
add = tf.add(a,b)
mul = tf.multiply(a,b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))