import os
import re
import sys
import time
from datetime import datetime

import tensorflow as tf

from model import network_model

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.optimizers import Adadelta, Adam, RMSprop


FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of images per batch")
tf.app.flags.DEFINE_integer('max_steps', 100000, "Number of steps to train")
tf.app.flags.DEFINE_string('job_dir', 'jobdir', "Checkpoints")
tf.app.flags.DEFINE_string('data_dir', 'data', "Checkpoints")
tf.app.flags.DEFINE_string('training_file', 'train.tfrecords', "Path to datasets")
tf.app.flags.DEFINE_string('validation_file', validation.tfrecords', "Path to datasets")
tf.app.flags.DEFINE_float('lr', 0.0005, 'Learning rate')
tf.app.flags.DEFINE_string('verbosity', 'INFO', "Control logging level")
tf.app.flags.DEFINE_integer('num_parallel_calls', 12, 'Input parallelization')
tf.app.flags.DEFINE_integer('throttle_secs', 120, "Evaluate every n seconds")

# Global constants describing the Tiny Imagenet data set.
IMAGE_SHAPE = [32, 32, 3]
INPUT_SHAPE = [None, 32, 32, 3]
INPUT_NAME = 'images'
NUM_CLASSES = 10 

# Define input pipelines


def scale_image(image):

    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
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
  image = tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0])
  
  if augment:
    # Pad 4 pixels on each dimension of feature map, done in mini-batch
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, IMAGE_SHAPE)
    image = tf.image.random_flip_left_right(image)
     
  label = features['label']
  label = tf.one_hot(label, NUM_CLASSES) 
  return image, label


def input_fn(filename, train, batch_size, num_parallel_calls):
  
  # Open a file
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.repeat(None if train else 1)
  # Parse records
  parse = lambda x: _parse(x, train)
  dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls)
  if train:
        dataset = dataset.shuffle(buffer_size=10000)
   
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=1)
  #
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
  
  return {INPUT_NAME: image_batch}, label_batch

def serving_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({INPUT_NAME: scaled_image}, {INPUT_NAME: input_image})


def train_evaluate():
  

  #Create an optimizer
  opt = Adam(lr = FLAGS.lr)
  
  #Create a keras model
  model = network_model(IMAGE_SHAPE, INPUT_NAME, NUM_CLASSES, opt)
  
  #Convert the the keras model to tf estimator
  estimator = model_to_estimator(keras_model = model, model_dir=FLAGS.job_dir)
  
  #Create training, evaluation, and serving input functions
  training_file = os.path.join(FLAGS.data_dir, FLAGS.training_file)
  validation_file = os.path.join(FLAGS.data_dir, FLAGS.validation_file)
  train_input_fn = lambda: input_fn(filename=training_file, train=True, batch_size=FLAGS.batch_size, num_parallel_calls = FLAGS.num_parallel_calls)
  valid_input_fn = lambda: input_fn(filename=validation_file, train=False, batch_size=FLAGS.batch_size, num_parallel_calls = FLAGS.num_parallel_calls)
  serving_input_fn = serving_fn

  #Create training and validation specifications
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.max_steps)
  
  export_latest = tf.estimator.FinalExporter("classifier", serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, 
                                      steps=None,
                                      throttle_secs=FLAGS.throttle_secs,
                                      exporters=export_latest)
  
  #Start training
  tf.logging.set_verbosity(FLAGS.verbosity)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  

def main(argv=None):
  
 if tf.gfile.Exists(FLAGS.job_dir):
   tf.gfile.DeleteRecursively(FLAGS.job_dir)
 tf.gfile.MakeDirs(FLAGS.job_dir)
  
 train_evaluate()
  

if __name__ == '__main__':
  tf.app.run()
  
  

    
    
    
