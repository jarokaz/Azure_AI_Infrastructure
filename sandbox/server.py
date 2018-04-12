import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('index', 0, "Task index")

def main(argv=None):
  jobs = {"local": ["localhost:2222", "localhost:2223"]}

  cluster = tf.train.ClusterSpec(jobs)
  config = tf.ConfigProto( log_device_placement=True)
  config.gpu_options.per_process_gpu_memory_fraction = 0.4

  print("starting server at:{0}".format(FLAGS.index))
  server = tf.train.Server(cluster, job_name="local", task_index=FLAGS.index, config = config)
  server.join()

if __name__ == '__main__':
  tf.app.run()
