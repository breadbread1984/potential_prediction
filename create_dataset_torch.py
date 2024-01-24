#!/usr/bin/python3

from absl import flags, app
from uuid import uuid1
from multiprocessing import Pool
from os import listdir, mkdir
from os.path import isdir, join, exists, splitext
from shutil import rmtree
import numpy as np
from torch.utils.data import Dataset, DataLoader

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('output_dir', default = 'dataset_torch', help = 'path to output directory')
  flags.DEFINE_list('eval_dists', default = ['1.7'], help = 'bond distances which are used as evaluation dataset')
  flags.DEFINE_integer('pool_size', default = 16, help = 'size of multiprocess pool')

def extract(npy_path, output_path):
  samples = np.load(npy_path)
  for sample in samples:
    potential = sample[3]
    density = np.reshape(sample[4:4+9**3], (9,9,9))
    grad_x = np.reshape(sample[4+9**3:4+(9**3)*2], (9,9,9))
    grad_y = np.reshape(sample[4+(9**3)*2:4+(9**3)*3], (9,9,9))
    grad_z = np.reshape(sample[4+(9**3)*3:4+(9**3)*4], (9,9,9))
    x = np.stack([density, grad_x, grad_y, grad_z], axis = 0)
    y = potential
    np.savez(join(output_path, '%s.npz' % str(uuid1())), x = x, y = y)

def main(unused_argv):
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  if exists(FLAGS.output_dir): rmtree(FLAGS.output_dir)
  mkdir(FLAGS.output_dir)
  mkdir(join(FLAGS.output_dir, 'train'))
  mkdir(join(FLAGS.output_dir, 'val'))
  pool = Pool(FLAGS.pool_size)
  handlers = list()
  for molecule in listdir(FLAGS.input_dir):
    if not isdir(join(FLAGS.input_dir, molecule)): continue
    for bond in listdir(join(FLAGS.input_dir, molecule)):
      stem, ext = splitext(bond)
      if ext != '.npy': continue
      distance = int(stem.replace('data_', ''))
      is_train_sample = True if distance not in eval_dists else False
      handlers.append(pool.apply_async(extract, (join(FLAGS.input_dir, molecule, bond), join(FLAGS.output_dir, 'train') if is_train_sample else join(FLAGS.output_dir, 'val'))))
  [handler.wait() for handler in handlers]

class RhoDataset(Dataset):
  def __init__(self, dataset_dir):
    super(RhoDataset, self).__init__()
    self.file_list = list()
    for f in listdir(dataset_dir):
      stem, ext = splitext(f)
      if ext != '.npz': continue
      self.file_list.append(join(dataset_dir, f))
  def __len__(self):
    return len(self.file_list)
  def __getitem__(self, idx):
    data = np.load(self.file_list[idx])
    x, y = data['x'].astype(np.float32), np.expand_dims(data['y'], axis = -1).astype(np.float32)
    return x, y

if __name__ == "__main__":
  add_options()
  app.run(main)

