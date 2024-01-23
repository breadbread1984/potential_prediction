#!/usr/bin/python3

from absl import app, flags
from os import listdir
from os.path import isdir, join, exists, splitext
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory containing npy')

def preprocess(data):
  value = np.maximum(1e-32 * np.ones_like(data), np.abs(data))
  value = np.where(value >= 0, value, -value)
  return 1 / value

def main(unused_argv):
  rho = list()
  rho_x = list()
  rho_y = list()
  rho_z = list()
  for sample in tqdm(listdir(join(FLAGS.input_dir, 'train'))):
    stem, ext = splitext(sample)
    if ext != '.npz': continue
    npz_path = join(FLAGS.input_dir, 'train', sample)
    data = np.load(npz_path)
    x = data['x'].astype(np.float32)
    rho.extend(preprocess(x[0].flatten().tolist()))
    rho_x.extend(preprocess(x[1].flatten().tolist()))
    rho_y.extend(preprocess(x[2].flatten().tolist()))
    rho_z.extend(preprocess(x[3].flatten().tolist()))

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
