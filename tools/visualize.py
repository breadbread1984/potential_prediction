#!/usr/bin/python3

from absl import flags, app
from os import listdir
from os.path import isdir, join, exists, splitext
import numpy as np
import tensorflow as tf
from models_tf import PredictorSmall
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_list('eval_dists', default = ['1.7',], help = 'bond distances which are used as evaluation dataset')
  flags.DEFINE_integer('groups', default = 1, help = 'group number for conv')

def set_configs():
  [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices('GPU')]

def main(unused_argv):
  if not exists(FLAGS.ckpt):
    raise Exception('checkpoint not found!')
  set_configs()
  predictor = PredictorSmall(in_channel = 4, groups = FLAGS.groups)
  optimizer = tf.keras.optimizers.Adam(1e-2)
  checkpoint = tf.train.Checkpoint(model = predictor, optimizer = optimizer)
  checkpoint.restore(tf.train.latest_checkpoint(join(FLAGS.ckpt, 'ckpt')))
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  for molecule in listdir(FLAGS.input_dir):
    if not isdir(join(FLAGS.input_dir, molecule)): continue
    for bond in listdir(join(FLAGS.input_dir, molecule)):
      stem, ext = splitext(bond)
      if ext != '.npy': continue
      distance = int(stem.replace('data_',''))
      is_eval_sample = True if distance in eval_dists else False
      if not is_eval_sample: continue
      print('plotting %s' % join(FLAGS.input_dir, molecule, bond))
      # verify convergence
      npy_path = join(FLAGS.input_dir, molecule, bond)
      samples = np.load(npy_path)
      coords = samples[:,1:3]
      idx = np.all(coords == 0, axis = -1)
      selected = samples[idx]
      new_idx = np.argsort(selected[:,0], axis = -1)
      selected = selected[new_idx]
      pred = list()
      gt = list()
      for sample in selected:
        density = np.reshape(sample[4:4+9**3], (9,9,9))
        grad_x = np.reshape(sample[4+9**3:4+(9**3)*2], (9,9,9))
        grad_y = np.reshape(sample[4+(9**3)*2:4+(9**3)*3], (9,9,9))
        grad_z = np.reshape(sample[4+(9**3)*3:4+(9**3)*4], (9,9,9))
        inputs = np.expand_dims(np.stack([density, grad_x, grad_y, grad_z], axis = -1), axis = 0)
        pred.append(predictor(inputs).numpy()[0])
        gt.append(sample[3])
      plt.cla()
      plt.plot(selected[:,0], gt, label = 'ground truth')
      plt.plot(selected[:,0], pred, label = 'prediction')
      plt.legend()
      plt.savefig('%s_%d.png' % (molecule, distance))

if __name__ == "__main__":
  add_options()
  app.run(main)
