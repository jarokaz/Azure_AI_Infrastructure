import os
import re
import sys
import time
from datetime import datetime

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of images per batch")
tf.app.flags.DEFINE_integer('epochs', 5, "Number of steps to train")
tf.app.flags.DEFINE_string('data_dir', '../cifar10-data', "Path to datasets")
tf.app.flags.DEFINE_string('job_dir', 'jobdir', "Checkpoints")
tf.app.flags.DEFINE_string('training_file', 'train.tfrecords', "Training file name")
tf.app.flags.DEFINE_string('validation_file', 'validation.tfrecords', "Validation file name")
tf.app.flags.DEFINE_integer('log_frequency', 50, 'How often to log results to the console.')
tf.app.flags.DEFINE_string('verbosity', 'INFO', "Control logging level")
tf.app.flags.DEFINE_float('lr', 1e-4, "Learning rate")
tf.app.flags.DEFINE_boolean('multi_gpu', False, "Utilize all GPUs")

### Define input pipelines

# Global constants describing the CIFAR-10 data set.
IMAGE_SHAPE = [32, 32, 3]
INPUT_SHAPE = [None, 32, 32, 3]
INPUT_NAME = 'images'
NUM_TRAIN_EXAMPLES = 50000
NUM_VALIDATION_EXAMPLES = 10000
NUM_CLASSES = 10


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
  #label = tf.one_hot(label, NUM_CLASSES, on_value=1, off_value=0)
  
  return image, label

def input_fn(filename, train, batch_size, buffer_size=10000):
  
  # Open a file
  dataset = tf.data.TFRecordDataset(filename)
  dataset = dataset.repeat(1)
  # Parse records
  parse = lambda x: _parse(x, train)
  if train:
        dataset = dataset.shuffle(buffer_size)
   
  dataset = dataset.map(parse)
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


### Define a network model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation


def simple_net(images, training):
  
  x = Conv2D(32, (3, 3), padding='same')(images)
  x = Activation('relu')(x)
  x = Conv2D(32, (3, 3))(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.25)(x, training=training)

  x = Conv2D(64, (3, 3), padding='same')(x)
  x = Activation('relu')(x)
  x = Conv2D(64, (3, 3))(x)
  x = Activation('relu')(x)
  x = MaxPooling2D(pool_size=(2, 2))(x)
  x = Dropout(0.25)(x, training=training)

  x = Flatten()(x)
  x = Dense(512)(x)
  x = Activation('relu')(x)
  x = Dropout(0.5)(x, training=training)
  logits = Dense(10)(x)
  
  return logits

    
### Define a model function for a custom estimator
def model_fn(features, labels, mode, params):
  
  if isinstance(features, dict):
    images = features[INPUT_NAME]
  else:
    images = features
    
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    if params.get('multi_gpu'):
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
      
    logits = simple_net(images, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
    
    # Name tensors to be logged with LoggingTensorHook
    tf.identity(FLAGS.lr, 'learning_rate')
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')
    # Save accuracy scalar to Tensorboard output
    tf.summary.scalar('train_accuracy', accuracy[1])
    
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      loss=loss,
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
  
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = simple_net(images, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      loss=loss,
      eval_metric_ops={'accuracy': 
                       tf.metrics.accuracy(
                         labels=labels, 
                         predictions=tf.argmax(logits, axis=1))})
    
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = simple_net(images, training=False)
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
        

def validate_batch_size_for_multi_gpu(batch_size):
  """
  For multi-gpu, batch-size must be a multiple of the number of GPUs.
  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.
  """
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)
  
 
### Define and start a training job
def train_evaluate():
  
  model_function = model_fn
  
  if FLAGS.multi_gpu:
    print("Training on multiple GPUs")
    validate_batch_size_for_multi_gpu(FLAGS.batch_size)
    model_function = tf.contrib.estimator.replicate_model_fn(
      model_fn, loss_reduction=tf.losses.Reduction.MEAN)
  

  cifar_classifier = tf.estimator.Estimator(
    model_fn=model_function, 
    model_dir=FLAGS.job_dir,
    params={
      'multi_gpu': FLAGS.multi_gpu
    })
  
  #Create training, evaluation, and serving input functions
  training_file = os.path.join(FLAGS.data_dir, FLAGS.training_file)
  validation_file = os.path.join(FLAGS.data_dir, FLAGS.validation_file)
  train_input_fn = lambda: input_fn(filename=training_file, batch_size=FLAGS.batch_size, train=True)
  valid_input_fn = lambda: input_fn(filename=validation_file, batch_size=FLAGS.batch_size, train=False)

  
  #Train and evaluate model
  for _ in range(FLAGS.epochs):
  #  cifar_classifier.train(input_fn=train_input_fn, hooks=train_hooks)
    cifar_classifier.train(input_fn=train_input_fn)
    eval_result = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print('\nEvaluation results: \n\t%s\n' % eval_results)
    
  

def main(argv=None):
  
 if tf.gfile.Exists(FLAGS.job_dir):
   tf.gfile.DeleteRecursively(FLAGS.job_dir)
 tf.gfile.MakeDirs(FLAGS.job_dir)
  
 train_evaluate()
  

if __name__ == '__main__':
  tf.app.run()
  
  

    
    
    