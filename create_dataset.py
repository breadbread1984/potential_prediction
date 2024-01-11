#!/usr/bin/python3

from absl import flags, app
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('output_tfrecord_prefix', default = '', help = 'name prefix for train and test set')

class Dataset(object):
  def __init__(self,):
    pass
  def get_parse_function(self,):
    def parse_function(serialized_example):
      feature = tf.io.parse_single_example(
        serialized_example,
        features = {
          'density': tf.io.FixedLenFeature((81,81,81,4), dtype = tf.float32),
          'potential': tf.io.FixedLenFeature((), dtype = tf.float32),
        })
      density = feature['density']
      potential = feature['potential']
      return density, potential
    return parse_function

def main(unused_argv):
  pass

if __name__ == "__main__":
  add_options()
  app.run(main)
