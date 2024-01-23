#!/usr/bin/python3

from absl import app, flags
from os import listdir
from os.path import isdir, join, exists, splitext
import numpy as np
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing npy')

def preprocess(data):
  value = np.maximum(1e-32 * np.ones_like(data), data)
  value = np.where(value >= 0, value, -value)
  return 1 / value

def main(unused_argv):
  rho = list()
  rho_x = list()
  rho_y = list()
  rho_z = list()
  for molecule in listdir(FLAGS.input_dir):
    if not isdir(join(FLAGS.input_dir, molecule)): continue
    for bond in listdir(join(FLAGS.input_dir, molecule)):
      stem, ext = splitext(bond)
      if ext != '.npy': continue
      npy_path = join(FLAGS.input_dir, molecule, bond)
      samples = np.load(npy_path)
      for sample in samples:
        rho.extend(preprocess(sample[4:4+9**3]).tolist())
        rho_x.extend(preprocess(sample[4+9**3:4+(9**3)*2]).tolist())
        rho_y.extend(preprocess(sample[4+(9**3)*2:4+(9**3)*3]).tolist())
        rho_z.extend(preprocess(sample[4+(9**3)*3:4+(9**3)*4]).tolist())
      break
  fig, axs = plt.subplots(2,2)
  axs[0,0].set_title('rho')
  axs[0,0].hist(rho, bins = 30, color = 'skyblue', alpha = 0.8)
  axs[0,1].set_title('drho/dx')
  axs[0,1].hist(rho_x, bins = 30, color = 'skyblue', alpha = 0.8)
  axs[1,0].set_title('drho/dy')
  axs[1,0].hist(rho_y, bins = 30, color = 'skyblue', alpha = 0.8)
  axs[1,1].set_title('drho/dz')
  axs[1,1].hist(rho_z, bins = 30, color = 'skyblue', alpha = 0.8)
  plt.savefig('distribution.png')
  plt.show()

if __name__ == "__main__":
  add_options()
  app.run(main)
