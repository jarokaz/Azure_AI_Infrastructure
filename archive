import os
import re
import sys
import time
from datetime import datetime

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of images per batch")
tf.app.flags.DEFINE_integer('max_steps', 500000, "Number of steps to train")
tf.app.flags.DEFINE_string('data_dir', '../../data/tinyimagenet', "Path to datasets")
tf.app.flags.DEFINE_integer('eval_steps', 1000, "Number of steps to train")
tf.app.flags.DEFINE_string('job_dir', 'jobdir', "Checkpoints")
tf.app.flags.DEFINE_string('training_file', 'train.tfrecords', "Training file name")
tf.app.flags.DEFINE_string('validation_file', 'validation.tfrecords', "Validation file name")
tf.app.flags.DEFINE_integer('log_frequency', 50, 'How often to log results to the console.')
tf.app.flags.DEFINE_string('verbosity', 'INFO', "Control logging level")
tf.app.flags.DEFINE_float('lr', 1e-4, "Learning rate")
tf.app.flags.DEFINE_boolean('multi_gpu', False, "Utilize all GPUs")
tf.app.flags.DEFINE_integer('num_parallel_calls', 12, "Parallelization of dataset.map")
tf.app.flags.DEFINE_integer('prefetch_buffer_size', 1, "Size of the prefetch buffer for dataset pipelines")
tf.app.flags.DEFINE_float('weight_decay', 0.0005, "L2 regularization")
tf.app.flags.DEFINE_integer('throttle_secs', 120, "Evaluate every n seconds")

# Global constants describing the Tiny Imagenet data set.
IMAGE_SHAPE = [64, 64, 3]
INPUT_SHAPE = [None, 64, 64, 3]
INPUT_NAME = 'images'
NUM_CLASSES = 200 

# Define input pipelines
  
def get_index(data_dir, mode):
  
  path = os.path.join(data_dir,  mode + '.txt')
  with open(path, 'r') as index:
    examples = [line.strip().split(' ') for line in index]

  images = [os.path.join(data_dir, 'images', example[0]) for example in examples]
  labels = [int(example[1]) for example in examples]
  
  return images, labels 

def scale_image(image):

    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
    return image

def process_image(augment):
  def _process_image(image_path, label): 
    image_string = tf.read_file(image_path) 
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.reshape(image_decoded, IMAGE_SHAPE)
    image = tf.cast(image, tf.float32)
    image = scale_image(image)
    
    if augment:
      image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SHAPE[0] + 8, IMAGE_SHAPE[1] + 8)
      image = tf.random_crop(image, IMAGE_SHAPE)
      image = tf.image.random_flip_left_right(image)
     
    label = tf.cast(label, tf.int32)

    return image, label
  
  return _process_image

  
def input_fn(data_dir, train, batch_size, num_parallel_calls):
 
  # Get a list of files 
  if train:
    images, labels = get_index(data_dir, 'training') 
  else:
    images, labels = get_index(data_dir, 'validation' )
  
  assert images != []
  assert len(images) == len(labels)
  
  # Cache, and shuffle, a list of files
  dataset = tf.data.Dataset.from_tensor_slices((images, labels)).repeat(None if train else 1)
 
  if train:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=len(images))
  
  # Parse records
  dataset = dataset.map(process_image(train), num_parallel_calls=num_parallel_calls)
  
  # Batch, prefetch, and serve
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=1)
  
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
  
  return {INPUT_NAME: image_batch}, label_batch

def serving_input_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({INPUT_NAME: scaled_image}, {INPUT_NAME: input_image})


### Define a model function for a custom estimation


#from resnet import model
from resnet import model

def model_fn(features, labels, mode, params):
  
  images = features[INPUT_NAME]
  network_model = model 

  if mode == tf.estimator.ModeKeys.TRAIN:
      tf.keras.backend.set_learning_phase(1)
  else:
      tf.keras.backend.set_learning_phase(0)
    
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    if params.get('multi_gpu'):
      print("Running on multiple GPUs")
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
      
    logits = network_model(images)
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
    logits = network_model(images)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.EVAL,
      loss=loss,
      eval_metric_ops={'accuracy': 
                       tf.metrics.accuracy(
                         labels=labels, 
                         predictions=tf.argmax(logits, axis=1))})
    
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = network_model(images)
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
  return model_fn

          

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

  classifier = tf.estimator.Estimator(
    model_fn=model_function, 
    model_dir=FLAGS.job_dir,
    params={
      'multi_gpu': FLAGS.multi_gpu
    })
  
  #Create training, evaluation, and serving input functions
  train_input_fn = lambda: input_fn(data_dir=FLAGS.data_dir, train=True, batch_size=FLAGS.batch_size, num_parallel_calls=FLAGS.num_parallel_calls)
  valid_input_fn = lambda: input_fn(data_dir=FLAGS.data_dir, train=False, batch_size=FLAGS.batch_size, num_parallel_calls=FLAGS.num_parallel_calls)
  
  #Create training and validation specifications
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.max_steps)
  
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
  
  
  

    
    
    
