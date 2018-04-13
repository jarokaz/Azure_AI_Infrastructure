import tensorflow as tf
import numpy as np
from time import strftime, time 
from os.path import join

import cifar10

file = '../cifar-10-data/eval.tfrecords'

with tf.Session() as sess:
  image_batch, label_batch = cifar10.get_batch(file, train=False, batch_size=2)
  result = sess.run([image_batch, label_batch])
  print(result[0].shape)
  print(result[1].shape)