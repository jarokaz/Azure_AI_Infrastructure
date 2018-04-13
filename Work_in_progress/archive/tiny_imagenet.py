import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Global constants describing the Tiny Imagenet data set.
IMAGE_SHAPE = [64, 64, 3]
CROPPED_IMAGE_SHAPE = [56, 56, 3]
INPUT_SHAPE = [None, 56, 56, 3]
INPUT_NAME = 'images'
NUM_CLASSES = 200 

 
def scale_image(image):

    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
    return image

def process_image(augment):
  def _process_image(image_path, label): 
    image_string = tf.read_file(image_path) 
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = scale_image(image)
    
    if augment:
      image = tf.random_crop(image, CROPPED_IMAGE_SHAPE)
      image = tf.image.random_flip_left_right(image)
    else:
      offset_width = (IMAGE_SHAPE[0] - CROPPED_IMAGE_SHAPE[0])//2
      offset_height = offset_width
      target_width = CROPPED_IMAGE_SHAPE[0]
      target_height = target_width
      image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
     
    label = tf.cast(label, tf.int32)

    return image, label
  
  return _process_image

  
def input_fn(data_dir, train=True, batch_size=32):
 
  # Get a list of files 
  images, labels = get_images_and_labels(data_dir, train) 
  
  assert images != []
  assert len(images) == len(labels)

  # Cache, and shuffle, a list of files
  dataset = tf.data.Dataset.from_tensor_slices((images, labels))
  dataset = dataset.cache()
  dataset = dataset.repeat(None if train else 1)

  if train:
    dataset = dataset.shuffle(buffer_size=len(images))
  
  # Parse records
  dataset = dataset.map(process_image(train), num_parallel_calls=FLAGS.num_parallel_calls)
  
  # Batch, prefetch, and serve
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
  image_batch, label_batch = iterator.get_next()
  
  return {INPUT_NAME: image_batch}, label_batch

def serving_input_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({INPUT_NAME: scaled_image}, {INPUT_NAME: input_image})

import glob
def get_images_and_labels(data_dir, train=True):
    
  with open(os.path.join(data_dir, 'wnids.txt')) as f:
    classes = [l.strip() for l in f]
    
  classes_to_labels = {class_name: label for label, class_name in enumerate(classes)}
  
  dataset = []
  if train:
    path = os.path.join('train', '*', 'images', '*.JPEG')
    images = glob.glob(os.path.join(data_dir, path))
    labels = [classes_to_labels[image[image.find('images')+7:image.rfind('_')]] for image in images]
  else:
    with open(os.path.join(data_dir, 'val', 'val_annotations.txt')) as f:
      records = [l.strip() for l in f]
                                
    path = os.path.join(data_dir, 'val', 'images')
    images = [os.path.join(path, record.split('\t')[0]) for record in records]                          
    labels = [classes_to_labels[record.split('\t')[1]] for record in records] 
    
    
  return images, labels 
