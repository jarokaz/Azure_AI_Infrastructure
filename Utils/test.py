import tensorflow as tf
import os

data_dir = '../data/tiny-imagenet'

def parse(example_proto):
  features = tf.parse_single_example(
        example_proto,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
  image = tf.decode_raw(features['image'], tf.uint8)
  #image = tf.reshape(image, [64, 64, 3])
  label = features['label']
  return image, label


def input_fn(data_dir, data_file):
  dataset = tf.data.TFRecordDataset(os.path.join(data_dir, data_file)).repeat(1)
  dataset = dataset.map(parse)
  dataset = dataset.batch(100)
  iterator = dataset.make_one_shot_iterator()
  image, label = iterator.get_next()
  return image, label

image, label = input_fn(data_dir, 'training_10.tfrecords')
with tf.Session() as sess:
    i = 1
    while True:
      i += 1
      images, labels = sess.run([image, label])  
      print("Batch: {0}, batch shape: {1}".format(i, images.shape))
       
        

