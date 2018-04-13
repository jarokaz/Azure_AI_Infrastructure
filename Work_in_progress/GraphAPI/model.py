import os
import re
import sys

import tensorflow as tf

from cifar10 import NUM_CLASSES, IMAGE_SHAPE


def _activation_summary(x):
  """Helper function to add tensorboard summaries for activations"""
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
  
   
  
 
  
def network_graph(images):
  """Builds an inference graph"""
  
  # First convolution layer
  with tf.variable_scope('conv1') as scope:
    kernel = tf.get_variable('weights', [5, 5, 3, 32])
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [32], initializer=tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)
    
  # Second convolution layer
  with tf.variable_scope('conv2') as scope:
    kernel = tf.get_variable('weights', [5, 5, 32, 64])
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)
    
  # Pooling layer
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1, 2, 2, 1], 
                         strides=[1, 2, 2, 1],
                         padding='VALID', name='pool1')
  
  # Dropout
  drop1 = tf.nn.dropout(pool1, keep_prob=0.5, name='drop1')
  
  # Flatten the tensor and add a dense layer
  with tf.variable_scope('dense1') as scope:
    image_size = drop1.shape[1].value * drop1.shape[2].value * drop1.shape[3].value
    reshape = tf.reshape(drop1, [tf.shape(drop1)[0], image_size])
    dim = reshape.shape[1].value
    weights = tf.get_variable('weights', [dim, 512])
    biases = tf.get_variable('biases', [512], initializer=tf.constant_initializer(0.0))
    dense1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(drop1)
    
  # Add output layer
  with tf.variable_scope('output') as scope:
    weights = tf.get_variable('weights', [512, NUM_CLASSES])
    biases = tf.get_variable('biases', [NUM_CLASSES], initializer=tf.constant_initializer(0.0))
    output = tf.add(tf.matmul(dense1, weights), biases, name=scope.name)
    _activation_summary(output)
    
  return output


  
  
  
    
  
  