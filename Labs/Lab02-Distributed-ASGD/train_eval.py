import os
import re
import sys
import time
from datetime import datetime
import json

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
  label = tf.one_hot(label, NUM_CLASSES)
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
    
  # Prefetch a batch at a time
  dataset = dataset.prefetch(buffer_size=batch_size)
    
  # Shuffle the records
  if is_training:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
  
  dataset = dataset.repeat(None if is_training else 1)
    
  # Parse records
  parse = lambda x: _parse(x, is_training)
  dataset = dataset.map(parse, num_parallel_calls=num_parallel_calls)
  
  # Batch, prefetch, and serve
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=1)
  
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
  
  return {INPUT_NAME: image_batch}, label_batch

def serving_input_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    image = tf.image.crop_to_bounding_box(image, OFFSET_HEIGHT, OFFSET_WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({INPUT_NAME: scaled_image}, {INPUT_NAME: input_image})



from resnet import network_model

def train_evaluate():

  
  #Create a keras model
  model = network_model(CROPPED_IMAGE_SHAPE, INPUT_NAME, NUM_CLASSES)
  loss = 'categorical_crossentropy'
  metrics = ['accuracy']
  opt = Adam(lr = FLAGS.lr)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  #Convert the the keras model to tf estimator
  estimator = model_to_estimator(keras_model = model, model_dir=FLAGS.job_dir)
    
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
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  
def fix_tf_config():
  """ Azure Batch AI does not properly set TF_CONFIG environment variable
  as required by new versions of Tensorflow and train_and_evaluate function 
  used in this solution. This function modifies TF_CONFIG so it works with
  train_and_evaluate
  """
    
  tf_config = json.loads(os.environ["TF_CONFIG"])
  cluster = tf_config['cluster']
  task = tf_config['task']
  worker = cluster['worker']
  chief = {"chief": [worker.pop(0)]} 
  cluster.update(chief)
  if task['type'] == 'master':
    task['type']='chief'
  elif task['type'] == 'worker':
    task['index'] = task['index'] - 1
  tf_config = {'cluster': cluster, 'task': task}
  os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(argv=None):
 
  fix_tf_config()  

  train_evaluate()

  
if __name__ == '__main__':
  tf.app.run()
  
  

    
    
    
