import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
IMAGE_SHAPE = [32, 32, 3]
INPUT_SHAPE = [None, 32, 32, 3]
INPUT_NAME = 'images'
NUM_TRAIN_EXAMPLES = 45000
NUM_VALIDATION_EXAMPLES = 10000
NUM_TESTING_EXAMPLES = 5000

#def scale_image(image):
#
#    """Scales image pixesl between -1 and 1"""
#    image = image / 127.5
#    image = image - 1.
#    return image

def scale_image(image):

    """Scales image pixesl between 0 and 1"""
    image = image / 255
    
    return image


def _parse(example_proto, augment):
  features = tf.parse_single_example(
        example_proto,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
  image = tf.decode_raw(features['image'], tf.uint8)
  image = tf.cast(image, tf.float32)
  image = scale_image(image)
  image = tf.reshape(image, IMAGE_SHAPE)
  
  if augment:
      # Pad 4 pixels on each dimension of feature map, done in mini-batch
      image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
      image = tf.random_crop(image, IMAGE_SHAPE)
      image = tf.image.random_flip_left_right(image)

     
  label = features['label']
  label = tf.one_hot(label, NUM_CLASSES)
  
  return image, label

def input_fn(filename, train, batch_size, buffer_size=10000):
  
  if train:
    rep = None 
    augment = True
  else:
    rep = 1
    augment = False

  # Open a file
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.repeat(rep)
  # Parse records
  parse = lambda x: _parse(x, augment)
  dataset = dataset.map(parse)
  if train:
        dataset = dataset.shuffle(buffer_size)
      
  dataset = dataset.batch(batch_size)
  #
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
  
  return {INPUT_NAME: image_batch}, label_batch

def serving_input_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({INPUT_NAME: scaled_image}, {INPUT_NAME: input_image})

