#!/usr/bin/python3

from os.path import exists, join
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
  flags.DEFINE_integer('decay_steps', default = 1000, help = 'decay steps')
  flags.DEFINE_integer('warmup_steps', default = 1000, help = 'warmup steps')
  flags.DEFINE_integer('batch_size', default = 128, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 600, help = 'epochs to train')

def main(unused_argv):
  uniformer = UniformerSmall(in_channel = 4, out_channel = FLAGS.channels, groups = FLAGS.groups)
  trainer = Trainer(uniformer)
  if exists(FLAGS.ckpt): trainer.load_weight(join(FLAGS.ckpt, 'variables', 'variables'))
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(0., decay_steps = FLAGS.decay_steps, warmup_steps = FLAGS.warmup_steps, warmup_target = 0.1))
  trainer.compile(optimizer = optimizer, loss = [tf.keras.losses.MeanAbsoluteError()], metrics = [tf.keras.metrics.MeanAbsoluteError()])
  trainset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'trainset.tfrecord')).map(Dataset.get_parse_function()).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  valset = tf.data.TFRecordDataset(join(FLAGS.dataset, 'valset.tfrecord')).map(Dataset.get_parse_function()).prefetch(FLAGS.batch_size).shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = join(FLAGS.ckpt, 'ckpt'), save_freq = FLAGS.save_freq),
  ]
  trainer.fit(trainset, epochs = FLAGS.epochs, validation_data = valset, callbacks = callbacks)

if __name__ == "__main__":
  add_options()
  app.run(main)
