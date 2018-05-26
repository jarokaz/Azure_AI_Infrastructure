import os
import urllib
import zipfile
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import random

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
 
tf.app.flags.DEFINE_string('data_dir', '../../data/aerial', "Directory with training and testing images")



def main(argv=None):
  
  # Map class labels to integers
  class_to_label = {'Barren':0, 'Cultivated':1, 'Developed':2, 'Forest':3, 'Herbaceous':4, 'Shrub':5}

  # Create training  labels
  print("Creating training label file")
  path = os.path.join(FLAGS.data_dir, 'train')
  with open(os.path.join(path,  'train_labels.txt'), 'w') as file:
    list_dir =  [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for folder in list_dir:
      for image in os.listdir(os.path.join(path, folder)):
        file.write("{0},{1}\n".format( image, class_to_label[folder]))

   # Create validation  labels
  print("Creating validation label file")
  path = os.path.join(FLAGS.data_dir, 'test')
  with open(os.path.join(path,  'test_labels.txt'), 'w') as file:
    list_dir =  [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for folder in list_dir:
      for image in os.listdir(os.path.join(path, folder)):
        file.write("{0},{1}\n".format( image, class_to_label[folder]))
  
  print('Done!')
 

if __name__ == '__main__':
  tf.app.run()
  
