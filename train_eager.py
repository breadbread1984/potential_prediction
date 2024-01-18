#!/usr/bin/python3

from os import listdir, mkdir
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
  uniformer = UniformerSmall(in_channel = 4, out_channel = FLAGS.channels, groups = FLAGS.groups)
  trainer = Trainer(uniformer)
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecayRestarts(FLAGS.lr, first_decay_steps = FLAGS.decay_steps))

  train_list, val_list = search_datasets(FLAGS.dataset)
  trainset = tf.data.TFRecordDataset(train_list).map(Dataset.get_parse_function()).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  valset = tf.data.TFRecordDataset(val_list).map(Dataset.get_parse_function()).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)

  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  checkpoint = tf.train.Checkpoint(model = trainer, optimizer = optimizer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))
  
  log = tf.summary.create_file_writer(FLAGS.ckpt)

  for epoch in range(FLAGS.epochs):
    # train
    train_metric = tf.keras.metrics.Mean(name = 'loss')
    train_iter = iter(trainset)
    for sample, label in train_iter:
      with tf.GradientTape() as tape:
        pred = trainer(sample)
        loss = tf.keras.losses.MeanAbsoluteError()(label, pred)
      train_metric.update_state(loss)
      grads = tape.gradient(loss, trainer.trainable_variables)
      optimizer.apply_gradients(zip(grads, trainer.trainable_variables))
      print('Step #%d loss: %f' % (optimizer.iterations, train_metric.result()))
      if optimizer.iterations % FLAGS.save_freq == 0:
        checkpoint.save(join(FLAGS.ckpt, 'ckpt'))
    # evaluate
    eval_metric = tf.keras.metrics.MeanAbsoluteError(name = 'MAE')
    eval_iter = iter(valset)
    for sample, label in eval_iter:
      pred = trainer(sample)
      eval_metric.update_state(label, pred)
      with log.as_default():
        tf.summary.scalar('mean absolute error', eval_metric.result(), step = optimizer.iterations)
      print('Step #%d MAE: %f' % (optimizer.iterations, eval_metric.result()))

  checkpoint.save(FLAGS.ckpt)

if __name__ == "__main__":
  add_options()
  app.run(main)
