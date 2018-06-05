# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:41:39 2017

@author: hrothe

@description: Introductory example for simple regression implemented in Tensorflow (see https://www.tensorflow.org/get_started/get_started)
"""

import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32) #weights
b = tf.Variable([-.3], dtype=tf.float32) #bias (aka intercept)
# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

# loss function
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate and print training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("Weight: %s bias: %s loss: %s"%(curr_W, curr_b, curr_loss))