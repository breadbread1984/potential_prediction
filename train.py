#!/usr/bin/python3

from os.path import exists, join
from absl import app, flags
import tensorflow as tf
from models import Trainer, UniformerSmall
from create_dataset import Dataset

FLAGS = flags.FLAGS

def add_option():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing train and test set')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('channels', default = 768, help = 'output channel')
  flags.DEFINE_integer('groups', default = 4, help = 'group number for conv')
  flags.DEFINE_integer('decay_steps', default = 1000, help = 'decay steps')
  flags.DEFINE_integer('warmup_steps', default = 1000, help = 'warmup steps')

def main(unused_argv):
  uniformer = UniformerSmall(in_channel = 4, out_channel = FLAGS.channels, groups = FLAGS.groups)
  trainer = Trainer(uniformer)
  if exists(FLAGS.ckpt): trainer.load_weight(join(FLAGS.ckpt, 'variables', 'variables'))
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(0, decay_steps = FLAGS.decay_steps, warmup_steps = FLAGS.warmup_steps, warmup_target = 0.1))
  trainer.compile(optimizer = optimizer, loss = [tf.keras.losses.MeanAbsoluteError], metrics = [tf.keras.metrics.MeanAbsoluteError])
  # TODO
