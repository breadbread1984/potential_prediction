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
  flags.DEFINE_float('dist', default = 1e-3, help = 'vector distance threshold')
  flags.DEFINE_float('potential', default = 0.2, help = 'potential distance threshold')

def main(unused_argv):
  index = IndexFlatL2(9**3*4)
  if exists('samples.index'):
    index = read_index('samples.index')
    labels = np.load('labels.npy')
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
    np.save('labels.npy', labels)
  print('searching for conflicts')
  output = open('conflicts.txt', 'w')
  for f in tqdm(listdir(join(FLAGS.input_dir, 'train'))):
    stem, ext = splitext(f)
    if ext != '.npz': continue
    data = np.load(join(FLAGS.input_dir, 'train', f))
    dists, indices = index.search(np.expand_dims(data['x'].flatten().astype(np.float32), axis = 0), k = 5)
    close_indices = indices[0][np.logical_and(0 < dists[0], dists[0] < FLAGS.dist)]
    close_labels = labels[close_indices]
    conflict_indices = close_labels[np.abs(close_labels - data['y']) > FLAGS.potential]
    output.write(' '.join([str(idx) for idx in conflict_indices]) + '\n')
  output.close()

if __name__ == "__main__":
  add_options()
  app.run(main)

