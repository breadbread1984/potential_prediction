#!/usr/bin/python3

from os import listdir, mkdir
from os.path import isdir, join, exists, splitext
from shutil import rmtree
from absl import flags, app
from multiprocessing import Pool
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
  @staticmethod
  def sample_generator(npy_path):
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
  @staticmethod
  def write_tfrecord(npy_path, tfrecord):
      print("%s => %s" % (npy_path, tfrecord))
      writer = tf.io.TFRecordWriter(tfrecord)
      for x,y in Dataset.sample_generator(npy_path):
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
            'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
          }))
        writer.write(trainsample.SerializeToString())
      writer.close()
  def generate_tfrecords(self, input_dir, output_dir, eval_dists = [1700], pool_size = 16):
    pool = Pool(pool_size)
    train_count, val_count = 0, 0
    def write_tfrecord(npy_path, tfrecord):
      print(npy_path, tfrecord)
      writer = tf.io.TFRecordWriter(tfrecord)
      for x,y in self.sample_generator(npy_path):
        trainsample = tf.train.Example(features = tf.train.Features(
          feature = {
            'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
            'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.serialize_tensor(x).numpy()])),
          }))
        writer.write(trainsample.SerializeToString())
      writer.close()
    if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
    mkdir(FLAGS.output_dir)
    handlers = list()
    for molecule in listdir(input_dir):
      if not isdir(join(input_dir, molecule)): continue
      for bond in listdir(join(input_dir, molecule)):
        stem, ext = splitext(bond)
        if ext != '.npy': continue
        distance = int(stem.replace('data_', ''))
        # decide whether to write to trainset or valset
        is_train_sample = True if distance not in eval_dists else False
        tfrecord_path = join(FLAGS.output_dir, ('trainset_%d.tfrecord' if is_train_sample else 'valset_%d.tfrecord') % (train_count if is_train_sample else val_count))
        train_count = (train_count + 1) if is_train_sample else train_count
        val_count = (val_count + 1) if not is_train_sample else val_count
        #write_tfrecord(join(input_dir, molecule, bond), tfrecord_path)
        handlers.append(pool.apply_async(Dataset.write_tfrecord, (join(input_dir, molecule, bond), tfrecord_path)))
    [handler.wait() for handler in handlers]
  @staticmethod
  def get_parse_function():
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
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  dataset = Dataset()
  dataset.generate_tfrecords(FLAGS.input_dir, FLAGS.output_dir, eval_dists = eval_dists)

if __name__ == "__main__":
  add_options()
  app.run(main)
