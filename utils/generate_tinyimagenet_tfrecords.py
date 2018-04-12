# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.
Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import urllib
import zipfile
from skimage.io import imread
import numpy as np
import random

import tensorflow as tf

TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
TINY_IMAGENET = 'tiny-imagenent-200.zip'

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_string('data_dir', '../data/tiny-imagenet', "Output dir")
 
def download_and_extract(data_dir):
  
  assert os.path.isdir(data_dir)
  
  filepath = os.path.join(data_dir, TINY_IMAGENET)
  
  if not os.path.exists(filepath):
    print("Downloading {0} to {1}".format(TINY_IMAGENET_URL, filepath))
    urllib.request.urlretrieve(TINY_IMAGENET_URL, filepath)
  else:
    print("{0} already downloaded".format(filepath))
  
  if not os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200')):
    print("Extracting files to {0}".format('tiny-imagenet-200'))
    with zipfile.ZipFile(os.path.join(data_dir, TINY_IMAGENET), "r") as zip_file:
      zip_file.extractall(data_dir)
    
  return os.path.join(data_dir, 'tiny-imagenet-200')
    

def convert_to_tfrecord(dataset, output_file):
  """Converts a file to TFRecords."""
  def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for record in dataset:
      image = imread(record[0])
      example = tf.train.Example(features=tf.train.Features(
        feature={
          'image': _bytes_feature(image.tobytes()),
          'label': _int64_feature(record[1])
        }))
      record_writer.write(example.SerializeToString())

                           
def main(argv=None):
  
  def map_labels(input_dir):
    wnids = os.path.join(input_dir, 'wnids.txt')
    assert os.path.exists(wnids)
    with open(wnids) as f:
      classes = [l.strip() for l in f]
    classes_to_labels = {class_name: label for label, class_name in enumerate(classes)}
    return classes_to_labels

  # Download and extract
  root_folder = download_and_extract(FLAGS.data_dir)
                   
  # Create numeric labels from orginal string ones
  print("Writing text to numeric mapping to: {0}".format('labels.txt'))
  class_to_labels = map_labels(root_folder)
  with open(os.path.join(FLAGS.data_dir, 'labels.txt'), 'w') as f:
    for key, value in class_to_labels.items():
      f.write("{0} {1}\n".format(key, value))
                   
  # Create a list of training images with labels
  dataset = []
  for folder in os.listdir(os.path.join(root_folder, 'train')):
    for image in os.listdir(os.path.join(root_folder, 'train', folder, 'images')):
      dataset.append(
        (os.path.join(root_folder, 'train', folder, 'images', image),
        class_to_labels[image[0: image.rfind('_')]]))
         
  # Shuffle the list
  random.shuffle(dataset)
  
  # Create ten tfrecord files
  num_shards = 10
  name_prefix = "training_"
  name_suffix = ".tfrecords"
  shard_size = len(dataset)//10
  
  for i in range(num_shards):
    filename = name_prefix + str(i+1) + name_suffix
    filename = os.path.join(FLAGS.data_dir, filename)
    if not os.path.exists(filename):
      convert_to_tfrecord(dataset[i*shard_size: (i+1)*shard_size], filename) 
  
  # Create a list of validation images with labels
  dataset = []
  with open(os.path.join(root_folder, 'val', 'val_annotations.txt'), 'r')  as f:
    for line in f:
      filename, classname, _, _, _, _ = line.split('\t') 
      dataset.append(
        (os.path.join(root_folder, 'val', 'images', filename),
        class_to_labels[classname]))
         
  # Shuffle the list
  random.shuffle(dataset)
  
  # Create ten tfrecord files
  filename = "validation.tfrecords" 
  filename = os.path.join(FLAGS.data_dir, filename)
  if not os.path.exists(filename):
    convert_to_tfrecord(dataset, filename) 
  
  print('Done!')
 

if __name__ == '__main__':
  tf.app.run()
  
  