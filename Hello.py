"""
HelloWorld example using TensorFlow library.
Author: Dilek Celik
"""
# Simple hello world using TensorFlow

from __future__ import print_function


import tensorflow as tf

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(hello))