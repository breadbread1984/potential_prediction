#!/usr/bin/python3

from absl import app, flags
from os import listdir
from os.path import splitext, join, exists
from tqdm import tqdm
import numpy as np
from faiss import write_index, read_index, IndexFlatL2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to dataset')
  flags.DEFINE_float('dist', default = 0.5, help = 'vector distance threshold')
  flags.DEFINE_float('potential', default = 0.2, help = 'potential distance threshold')

def main(unused_argv):
  index = IndexFlatL2(64)
  if exists('samples.index'): read_index(index, 'samples.index')
  else:
    print('generating index')
    features = list()
    labels = list()
    for f in tqdm(listdir(join(FLAGS.input_dir, 'train'))):
      stem, ext = splitext(f)
      if ext != '.npz': continue
      data = np.load(join(FLAGS.input_dir, 'train', f))
      features.append(data['x'].flatten().astype(np.float32))
      labels.append(data['y'].astype(np.float32))
    features = np.stack(features, axis = 0)
    labels = np.stack(labels, axis = 0)
    index.add(features)
    write_index(index, 'samples.index')
  print('searching for conflicts')
  for f in tqdm(listdir(FLAGS.input_dir)):
    stem, ext = splitext(f)
    if ext != '.npz': continue
    data = np.load(join(FLAGS.input_dir, f))
    index.search(data['x'].flatten().astype(np.float32), k = 5)
    break

if __name__ == "__main__":
  add_options()
  app.run(main)

