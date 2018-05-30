import os
import re
import sys
import time
from datetime import datetime

import tensorflow as tf

from tensorflow.python.keras import regularizers
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.optimizers import Adadelta, Adam, RMSprop

import horovod.tensorflow as hvd


FLAGS = tf.app.flags.FLAGS

# Default global parameters
tf.app.flags.DEFINE_integer('batch_size', 32, "Number of images per batch")
tf.app.flags.DEFINE_integer('max_steps', 100000, "Number of steps to train")
tf.app.flags.DEFINE_string('job_dir', '../../jobdir/run1', "Checkpoints")
tf.app.flags.DEFINE_string('data_dir', '../../data/tiny-imagenet', "Data")
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate')
tf.app.flags.DEFINE_string('verbosity', 'INFO', "Control logging level")
tf.app.flags.DEFINE_integer('num_parallel_calls', 12, 'Input parallelization')
tf.app.flags.DEFINE_integer('throttle_secs', 120, "Evaluate every n seconds")
tf.app.flags.DEFINE_integer('hidden_units', 256, "Hidden units")
                           

from model import network_model
from feed import INPUT_NAME, IMAGE_SHAPE, NUM_CLASSES, input_fn, serving_input_fn

def model_fn(features, labels, mode, params):
      
  # Create a model
  images = features[INPUT_NAME]
  logits = network_model(FLAGS.hidden_units, images, NUM_CLASSES)  
      
  # Define a predict mode Estimator spec
  if mode == tf.estimator.ModeKeys.PREDICT: 
    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
    }
    return tf.estimator.EstimatorSpec(
      mode = tf.estimator.ModeKeys.PREDICT,
      predictions=predictions,
      export_outputs={
        'classify': tf.estimator.export.PredictOutput(predictions)
      })
  
  # Define a a loss function  
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Define a train mode Estimator spec
  if mode == tf.estimator.ModeKeys.TRAIN:
    
    # Define an optimizer and train op
    optimizer = tf.train.MomentumOptimizer(
            learning_rate=FLAGS.lr * hvd.size(), momentum=0.9)  
    
    # Add Horovod distributed optimizer
    
    optimizer = hvd.DistributedOptimizer(optimizer)
     
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_or_create_global_step())
    
    accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
    tf.identity(accuracy[1], name='train_accuracy')
    
    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      loss=loss,
      train_op=train_op)
  
  # Define an eval mode Estimator spec
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
    
  # Initialize Horovod
  hvd.init()
  
    
  # Pin GPU to be used to process local rank (one GPU per process)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.visible_device_list = str(hvd.local_rank())


  # Save checkpoint only on worker 0 to prevent other workers from corrupting
  model_dir = FLAGS.job_dir if hvd.rank() == 0 else None
  
  estimator = tf.estimator.Estimator(model_fn=model_fn, 
                                     model_dir=model_dir,
                                     config=tf.estimator.RunConfig(session_config=config))

  #Create training, evaluation, and serving input functions
  train_input_fn = lambda: input_fn(data_dir=FLAGS.data_dir, 
                                    is_training=True, 
                                    batch_size=FLAGS.batch_size, 
                                    num_parallel_calls=FLAGS.num_parallel_calls)
    
  valid_input_fn = lambda: input_fn(data_dir=FLAGS.data_dir, 
                                    is_training=False, 
                                    batch_size=FLAGS.batch_size, 
                                    num_parallel_calls=FLAGS.num_parallel_calls)
  
  
  
  
  tensors_to_log = {"train_accuracy": "train_accuracy"}
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100)
  
  bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
  
  hooks = [logging_hook, bcast_hook]
  
  # Train
  tf.logging.set_verbosity(FLAGS.verbosity)
  
  estimator.train(
    input_fn=train_input_fn,
    steps=1000 // hvd.size(),
    hooks=hooks
    )
  
  # Evaluate
  eval_results = estimator.evaluate(
    input_fn=eval_input_fn)
  
  print(eval_results)
  

def main(argv=None):
 
  train_evaluate()

if __name__ == '__main__':
  tf.app.run()
  
  

    
    
    
