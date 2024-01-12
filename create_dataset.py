#!/usr/bin/python3

from os import listdir
from os.path import isdir, join, exists, splitext
from absl import flags, app
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('output_dir', default = '', help = 'path to output directory')
  flags.DEFINE_list('eval_dists', default = ['1.7',], help = 'bond distances which are used as evaluation dataset')

class Dataset(object):
  def __init__(self,):
    pass
  def sample_generator(self, npy_path):
    samples = np.load(npy_path)
    for sample in samples:
      coordinate = sample[:3]
      potential = sample[3]
      density = np.reshape(sample[4:4+9**3], (9,9,9))
      grad_x = np.reshape(sample[4+9**3:4+(9**3)*2], (9,9,9))
      grad_y = np.reshape(sample[4+(9**3)*2:4+(9**3)*3], (9,9,9))
      grad_z = np.reshape(sample[4+(9**3)*3:4+(9**3)*4], (9,9,9))
      x, y = tf.constant(np.stack([density, grad_x, grad_y, grad_z], axis = -1), dtype = tf.float32), tf.constant(potential, dtype = tf.float32)
      assert x.shape == (9,9,9,4)
      yield x,y
  def generate_tfrecords(self, input_dir, output_dir, eval_dists = [1.7]):
    trainset = tf.io.TFRecordWriter(join(output_dir, 'trainset.tfrecord'))
    valset = tf.io.TFRecordWriter(join(output_dir, 'valset.tfrecord'))
    for molecule in listdir(input_dir):
      if not isdir(molecule): continue
      for bond in listdir(join(input_dir, molecule)):
        stem, ext = splitext(bond)
        if ext != '.npy': continue
        distance = float(stem.replace('dm_', ''))
        target = trainset if distance not in eval_dist else valset
        for x,y in self.sample_generator(join(input_dir, molecule, bond)):
          trainsample = tf.train.Example(features = tf.train.Features(
            feature = {
              'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
              'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
            }))
          target.write(trainsample.SerializeToString())
    trainset.close()
    valset.close()
  @classmethod
  def get_parse_function(self,):
    def parse_function(serialized_example):
      feature = tf.io.parse_single_example(
        serialized_example,
        features = {
          'x': tf.io.FixedLenFeature((), dtype = tf.string),
          'y': tf.io.FixedLenFeature((), dtype = tf.string),
        })
      x = tf.io.parse_tensor(feature['x'], out_type = tf.float32)
      assert x.shape == (9,9,9,4)
      y = tf.io.parse_tensor(feature['y'], out_type = tf.float32)
      assert y.shape == ()
      return x, tf.math.exp(y)
    return parse_function

def main(unused_argv):
  eval_dists = [float(d) for d in FLAGS.eval_dists]
  dataset = Dataset()
  dataset.generate_tfrecords(FLAGS.input_dir, FLAGS.output_dir, eval_dists = eval_dists)

if __name__ == "__main__":
  add_options()
  app.run(main)
