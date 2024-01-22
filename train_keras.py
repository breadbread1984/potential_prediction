#!/usr/bin/python3

from os import listdir
from os.path import exists, join, splitext
from absl import app, flags
import tensorflow as tf
from models_9 import Trainer, UniformerSmall
from create_dataset import Dataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing train and test set')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('channels', default = 768, help = 'output channel')
  flags.DEFINE_integer('groups', default = 1, help = 'group number for conv')
  flags.DEFINE_integer('batch_size', default = 128, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 600, help = 'epochs to train')
  flags.DEFINE_float('lr', default = 0.01, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 200000, help = 'decay steps')
  flags.DEFINE_boolean('dist', default = False, help = 'whether to use data parallelism')

def set_configs():
  [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU')]

def search_datasets(dataset_path):
  train_list, val_list = list(), list()
  for f in listdir(dataset_path):
    stem, ext = splitext(f)
    if ext != '.tfrecord': continue
    if stem.startswith('trainset'): train_list.append(join(dataset_path, f))
    if stem.startswith('valset'): val_list.append(join(dataset_path, f))
  return train_list, val_list

def main(unused_argv):
  set_configs()
  if FLAGS.dist:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      uniformer = UniformerSmall(in_channel = 4, out_channel = FLAGS.channels, groups = FLAGS.groups)
      trainer = Trainer(uniformer)
      optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecayRestarts(FLAGS.lr, first_decay_steps = FLAGS.decay_steps))
      loss = [tf.keras.losses.MeanAbsoluteError()]
      metrics = [tf.keras.metrics.MeanAbsoluteError()]
  else:
    uniformer = UniformerSmall(in_channel = 4, out_channel = FLAGS.channels, groups = FLAGS.groups)
    trainer = Trainer(uniformer)
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecayRestarts(FLAGS.lr, first_decay_steps = FLAGS.decay_steps))
    loss = [tf.keras.losses.MeanAbsoluteError()]
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
  if exists(FLAGS.ckpt): trainer.load_weights(join(FLAGS.ckpt, 'ckpt', 'variables', 'variables'))
  trainer.compile(optimizer = optimizer, loss = loss, metrics = metrics)
  train_list, val_list = search_datasets(FLAGS.dataset)
  if len(train_list) == 0 or len(val_list) == 0:
    raise Exception('no tfrecord files found!')
  batch_size = FLAGS.batch_size if not FLAGS.dist else FLAGS.batch_size * strategy.num_replicas_in_sync
  trainset = tf.data.TFRecordDataset(train_list).map(Dataset.get_parse_function()).prefetch(batch_size).shuffle(batch_size).batch(batch_size)
  valset = tf.data.TFRecordDataset(val_list).map(Dataset.get_parse_function()).prefetch(batch_size).shuffle(batch_size).batch(batch_size)
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.ckpt, 'ckpt'), save_freq = FLAGS.save_freq),
  ]
  trainer.fit(trainset, epochs = FLAGS.epochs, validation_data = valset, callbacks = callbacks)

if __name__ == "__main__":
  add_options()
  app.run(main)