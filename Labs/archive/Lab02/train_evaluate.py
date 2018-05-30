from skimage.io import imread
from skimage.transform import resize
import random
import numpy as np
import os

import tensorflow as tf

from tensorflow.python.keras.utils import Sequence, to_categorical
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta, Adam

from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input

import horovod.keras as hvd

class ImageSequence(Sequence):
    def __init__(self, dataset, batch_size, is_training):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.is_training = is_training
        self.class_to_label = {'Barren':0, 'Cultivated':1, 'Developed':2, 'Forest':3, 'Herbaceous':4, 'Shrub':5}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_images, batch_labels  = zip(*self.dataset[idx * self.batch_size: (idx + 1) * self.batch_size])
        batch_int_labels = [self.class_to_label[label] for label in batch_labels]
        batch_onehot_labels = to_categorical(batch_int_labels, NUM_CLASSES)
        
        return np.array([imread(file_name) for file_name in batch_images]), np.array(batch_onehot_labels)

    def on_epoch_end(self):
        if self.is_training:
            random.shuffle(self.dataset) 



IMAGE_SHAPE = (224, 224, 3)
INPUT_NAME = 'images'
NUM_CLASSES = 6


def network_model(hidden_units):
     
    inputs = Input(shape=IMAGE_SHAPE, name=INPUT_NAME)
    conv_base = ResNet50(weights='imagenet',
                      include_top=False,
                      input_tensor=inputs,
                      pooling = 'avg')
                
    for layer in conv_base.layers:
        layer.trainable = False

    a = Flatten()(conv_base.output)
    a = Dense(hidden_units, activation='relu')(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
                                 
    model = Model(inputs=inputs, outputs=y)
                                                           
    return model

def train_evaluate():

    # Generate training and validation data generators 
    def get_image_list(data_dir):
       dataset = []
       for folder in os.listdir(data_dir):
          for image in os.listdir(os.path.join(data_dir, folder)):
             dataset.append((os.path.join(data_dir, folder, image), folder)) 
       return dataset      

    training_data = ImageSequence(get_image_list(os.path.join(FLAGS.data_dir, 'train')), FLAGS.batch_size, True)
    validation_data = ImageSequence(get_image_list(os.path.join(FLAGS.data_dir, 'test')), FLAGS.batch_size, False)

    # Horovod: Initialize Horovod
    hvd.init()

    # Horvod: Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.keras.backend.set_session(tf.Session(config=config))

    # Create a model
    model = network_model(FLAGS.hidden_units)
    loss = 'categorical_crossentropy'

    # Horovod: Adjust learning rate based on number of GPUs
    optimizer = Adadelta(lr=1.0 * hvd.size())
    # Horovod: add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(optimizer)

    metrics = ['acc']
    model.compile(optimizer, loss, metrics)
  
    # Set up callbacks
    callbacks = [
        # Broadcast initial variable states from rank 0 to all other processes
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
    
    # Horovod: save  logs only on worker 0
    if hvd.rank() == 0:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=FLAGS.log_dir))

    # Start training
    model.fit_generator(generator = training_data,
                        validation_data = validation_data,
                        epochs = FLAGS.epochs,
                        use_multiprocessing = True,
                        workers = 4,
                        callbacks = callbacks,
                        verbose = 1)

    # Save the model
    model.save(FLAGS.save_model_path)

### Job configuration and start up
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../../../data/aerial', "Directory with training and testing images")
tf.app.flags.DEFINE_string('save_model_path', '../../../SaveModel/aerial.h5', 'Path to model file')
tf.app.flags.DEFINE_string('log_dir', '../../../logdir/aearial1', 'Path to checkpoints and logs')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Number of images per batch')
tf.app.flags.DEFINE_integer('epochs', 20, 'Number of epochs to train')
tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of hidden units in the top FCN layer')


def main(argv=None):
   
   if tf.gfile.Exists(FLAGS.log_dir):
       tf.gfile.DeleteRecursively(FLAGS.log_dir)
   tf.gfile.MakeDirs(FLAGS.log_dir)
                
   train_evaluate()


if __name__ == '__main__':
    main()



        

