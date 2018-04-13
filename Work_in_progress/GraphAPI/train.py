import os
import re
import sys
import time
from datetime import datetime

import tensorflow as tf

import model

import cifar10

FLAGS = tf.app.flags.FLAGS

# Default model parameters
tf.app.flags.DEFINE_integer('batch_size', 64, "Number of images per batch")
tf.app.flags.DEFINE_integer('max_steps', 5000, "Number of steps to train")
tf.app.flags.DEFINE_string('data_dir', '../cifar-10-data', "Path to datasets")
tf.app.flags.DEFINE_string('job_dir', 'jobdir', "Checkpoints")
tf.app.flags.DEFINE_string('training_file', 'train.tfrecords', "Training file name")
tf.app.flags.DEFINE_string('validation_file', 'validattion.tfrecords', "Validation file name")
tf.app.flags.DEFINE_integer('log_frequency', 50, 'How often to log results to the console.')


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _loss(output, labels):
  """Returns a loss function given network output and tensor of labels"""
  
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels,
    logits=output,
    name='cross_entropy_per_example')
  
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  
  return cross_entropy_mean

def _add_loss_summaries(loss):
  """Add tensorboard summaries for losses"""
  tf.summary.scalar(loss.op.name + '/loss', loss)
  
def _optimizer(lr):
  """Return optimizer"""
  
  opt = tf.train.GradientDescentOptimizer(lr)
  
  return opt

  
def get_train_op(images, labels, global_step):
  """Configures optimizer and returns the train op"""
  output = model.network_graph(images)
  loss = _loss(output, labels)
  _add_loss_summaries(loss)
  
  # Variables that affect learning rate
  num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  
  # Configure exponential learning rate decay based on the number of steps
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                 global_step,
                                 decay_steps,
                                 LEARNING_RATE_DECAY_FACTOR,
                                 staircase=True)
  
  tf.summary.scalar('learning_rate', lr)
  
  # Compute and apply gradients
  opt = _optimizer(lr)
  grads = opt.compute_gradients(loss)
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
  
  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')
    
  return train_op, loss
  
  
def train_evaluate():
  """Train the model on CIFAR-10 dataset"""
  
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    
    
    # Force the input pipeline to run on CPU:0
    with tf.device('/cpu:0'):
      images, labels = cifar10.get_batches(
        filename = os.path.join(FLAGS.data_dir, FLAGS.training_file),
        train = True,
        batch_size = FLAGS.batch_size)
      
    train_op, loss = get_train_op(images, labels, global_step)
    
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))
    
   
    with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.job_dir,
      hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
              tf.train.NanTensorHook(loss),
              _LoggerHook()],
      config=tf.ConfigProto(log_device_placement=True)) as mon_sess:
      
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
        

def main(argv=None):
  

  if tf.gfile.Exists(FLAGS.job_dir):
    tf.gfile.DeleteRecursively(FLAGS.job_dir)
  tf.gfile.MakeDirs(FLAGS.job_dir)
  
  train_evaluate()
  

if __name__ == '__main__':
  tf.app.run()
  
  

    
    
    