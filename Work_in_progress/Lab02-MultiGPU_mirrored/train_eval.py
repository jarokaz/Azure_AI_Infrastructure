import os
import re
import sys
import time
from datetime import datetime

import tensorflow as tf


from tensorflow.python.keras import regularizers
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.optimizers import Adadelta, Adam, RMSprop


FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of images per batch")
tf.app.flags.DEFINE_integer('max_steps', 100000, "Number of steps to train")
tf.app.flags.DEFINE_string('job_dir', '../../jobdir/run1', "Checkpoints")
tf.app.flags.DEFINE_string('data_dir', '../../data/tiny-imagenet', "Data")
tf.app.flags.DEFINE_float('lr', 0.0005, 'Learning rate')
tf.app.flags.DEFINE_string('verbosity', 'INFO', "Control logging level")
tf.app.flags.DEFINE_integer('num_parallel_calls', 12, 'Input parallelization')
tf.app.flags.DEFINE_integer('throttle_secs', 600, "Evaluate every n seconds")
                           
# Global constants describing the Tiny Imagenet data set.
INPUT_SHAPE = [None, 64, 64, 3]
IMAGE_SHAPE = [64, 64, 3]
CROPPED_IMAGE_SHAPE = [56, 56, 3]
OFFSET_HEIGHT = 4
OFFSET_WIDTH = 4
TARGET_HEIGHT = 56
TARGET_WIDTH = 56


INPUT_NAME = 'images'
NUM_CLASSES = 200 
NUM_TRAINING_FILES = 10 

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
  image = tf.reshape(image, IMAGE_SHAPE)
  
  if augment:
    image = tf.random_crop(image, CROPPED_IMAGE_SHAPE)
    image = tf.image.random_flip_left_right(image)
  else:
    image = tf.image.crop_to_bounding_box(image, OFFSET_HEIGHT, OFFSET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
     
  label = features['label']
  # label = tf.one_hot(label, NUM_CLASSES)
  return image, label


def get_filenames(is_training, data_dir):
    if is_training:
        files = [os.path.join(data_dir, "training_{0}.tfrecords".format(i+1)) for i in range(NUM_TRAINING_FILES)]
    else: 
        files = [os.path.join(data_dir, "validation.tfrecords")]
    return files
        
    
def input_fn(data_dir, is_training, batch_size, num_parallel_calls, shuffle_buffer=10000):
 
  files = get_filenames(is_training, data_dir)
  dataset = tf.data.Dataset.from_tensor_slices(files)
  
  # Shuffle the input files
  if is_training:
    dataset = dataset.shuffle(buffer_size=NUM_TRAINING_FILES)
   
  # Convert to individual records
  dataset = dataset.flat_map(tf.data.TFRecordDataset)
    
    
  # Shuffle the records
  if is_training:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
  
  dataset = dataset.repeat(None if is_training else 1)
    
  # Parse records
  parse = lambda x: _parse(x, is_training)
  dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls)
  
  # Batch, prefetch, and serve
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=None)
  
  return dataset

def serving_input_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    image = tf.image.crop_to_bounding_box(image, OFFSET_HEIGHT, OFFSET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({INPUT_NAME: scaled_image}, {INPUT_NAME: input_image})



from resnet import network_model

def model_fn(features, labels, mode):
  
  images = features
    
  if mode == tf.estimator.ModeKeys.TRAIN:
      tf.keras.backend.set_learning_phase(1)
  else:
      tf.keras.backend.set_learning_phase(0)
   
  logits = network_model(images, NUM_CLASSES)
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(
      mode = tf.estimator.ModeKeys.PREDICT,
      predictions=predictions,
      export_outputs={
        'classify': tf.estimator.export.PredictOutput(predictions)
      })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.lr).minimize(loss, tf.train.get_or_create_global_step())
    
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      loss=loss,
      train_op=train_op)
  
  if mode == tf.estimator.ModeKeys.EVAL:
    
    eval_accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
    eval_metric = {'eval_accuracy': eval_accuracy}
    
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      loss=loss,
      eval_metric_ops=eval_metric)
  
  return model_fn


def train_evaluate():


  #Convert the the keras model to tf estimator
  distribution = tf.contrib.distribute.MirroredStrategy()
  run_config = tf.estimator.RunConfig(train_distribute=distribution)
    
  classifier = tf.estimator.Estimator(model_fn=model_fn, config=run_config, model_dir=FLAGS.job_dir)
    
  #Create training, evaluation, and serving input functions
  train_input_fn = lambda: input_fn(data_dir=FLAGS.data_dir, 
                                    is_training=True, 
                                    batch_size=FLAGS.batch_size, 
                                    num_parallel_calls=FLAGS.num_parallel_calls)
    
  valid_input_fn = lambda: input_fn(data_dir=FLAGS.data_dir, 
                                    is_training=False, 
                                    batch_size=FLAGS.batch_size, 
                                    num_parallel_calls=FLAGS.num_parallel_calls)
  
  #Create training and validation specifications
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, 
                                      max_steps=FLAGS.max_steps)
  
  export_latest = tf.estimator.FinalExporter("classifier", serving_input_fn)
    
  eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, 
                                    steps=None,
                                    throttle_secs=FLAGS.throttle_secs,
                                    exporters=export_latest)
  
 
  #Start training
  tf.logging.set_verbosity(FLAGS.verbosity)
  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
  

def main(argv=None):
 
  if tf.gfile.Exists(FLAGS.job_dir):
    tf.gfile.DeleteRecursively(FLAGS.job_dir)
  tf.gfile.MakeDirs(FLAGS.job_dir)
  
  train_evaluate()
  

if __name__ == '__main__':
  tf.app.run()
  
  

    
    
    
