#!/usr/bin/python3

from absl import app, flags
from os import listdir
from os.path import splitext, join, exits
import tqdm
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to dataset')

def main(unused_argv):
  index = cv2.ml.KNearest_create()
  if exist('index.xml'): index.load('index.xml')
  else:
    index.setIsClassifier(True)
    index.setDefaultK(1)
    features = list()
    labels = list()
    for f in tqdm(listdir(FLAGS.input_dir)):
      stem, ext = splitext(f)
      if ext != '.npz': continue
      data = np.load(join(FLAGS.input_dir, f))
      features.append(data['x'].flatten().astype(np.float32))
      labels.append(data['y'].astype(np.float32))
    features = np.stack(features, axis = 0)
    labels = np.stack(labels, axis = 0)
    index.train(features, cv2.ml.ROW_SAMPLE, labels)
    index.save('index.xml')
  
