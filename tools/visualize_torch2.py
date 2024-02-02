#!/usr/bin/python3

from absl import flags, app
from os import listdir
from os.path import isdir, join, exists, splitext
import pickle
import numpy as np
import torch
from torch import load, device
from models_torch import PredictorSmall
import matplotlib.pyplot as plt
from matplotlib import cm

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to input directory')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_list('eval_dists', default = ['1.7',], help = 'bond distances which are used as evaluation dataset')
  flags.DEFINE_integer('groups', default = 1, help = 'group number for conv')

def main(unused_argv):
  if not exists(FLAGS.ckpt):
    raise Exception('checkpoint not found!')
  ckpt = load(join(FLAGS.ckpt, 'model.pth'))
  model = PredictorSmall(in_channel = 4, groups = FLAGS.groups)
  model.load_state_dict(ckpt['state_dict'])
  model.eval().to(device('cuda'))
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  colors = cm.rainbow(np.linspace(0,1,))
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
      fig = plt.figure()
      ax = fig.add_subplot(111, projection = '3d')
      results = list()
      minval = np.finfo(np.float32).max
      maxval = 0
      for sample in samples:
        x,y,z = sample[:3]
        vxc_gt = sample[3]
        density = np.reshape(sample[4:4+9**3], (9,9,9))
        grad_x = np.reshape(sample[4+9**3:4+(9**3)*2], (9,9,9))
        grad_y = np.reshape(sample[4+(9**3)*2:4+(9**3)*3], (9,9,9))
        grad_z = np.reshape(sample[4+(9**3)*3:4+(9**3)*4], (9,9,9))
        inputs = np.expand_dims(np.stack([density, grad_x, grad_y, grad_z], axis = 0), axis = 0)
        inputs = torch.from_numpy(inputs.astype(np.float32)).to(device('cuda'))
        vxc_pred = np.log(model(inputs).detach().cpu().numpy()[0,0])
        diff = np.abs(vxc_gt - vxc_pred)
        minval = diff if diff < minval else minval
        maxval = diff if diff > maxval else maxval
        results.append((diff, x, y, z))
      diff_range = max(maxval - minval,1e-8)
      for diff, x, y, z in results:
        color = cm.rainbow(diff / diff_range)
        ax.scatter(x, y, z, c = color)
      with open('%s_%d.pickle', 'wb') as f:
        pickle.dump(ax, f)

if __name__ == "__main__":
  add_options()
  app.run(main)
