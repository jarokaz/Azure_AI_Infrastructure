import tensorflow as tf

with tf.device("/job:local/task:0/device:CPU:0"):
  a = tf.constant(1.0)
  b = tf.constant(2.0)
  
with tf.device("/job:local/task:1"):
  c = a + b
  
  
with tf.Session("grpc://localhost:2223") as sess:
  sess.run(c)
  
