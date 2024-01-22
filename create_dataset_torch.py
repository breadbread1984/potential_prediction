#!/usr/bin/python3

from os import listdir
from os.path import isdir, join, exists, splitext
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RhoDataset(Dataset):
  def __init__(self, dataset_dir, eval_dists, train = True):
    super(RhoDataset, self).__init__()
    self.dataset_dir = dataset_dir
    self.eval_dists = eval_dists
    self.is_train = is_train
    self.file_list = list()
    self.sample_count = 0
    for molecule in listdir(self.dataset_dir):
      if not isdir(join(self.dataset_dir, molecule)): continue
      for bond in listdir(join(self.dataset_dir, molecule)):
        stem, ext = splitext(bond)
        if ext != '.npy': continue
        distance = int(stem.replace('data_', ''))
        is_train_sample = True if distance not in eval_dists else False
        if train == is_train_sample:
          data = np.load(join(self.dataset_dir, molecule, bond))
          self.sample_list((join(self.dataset_dir, molecule, bond), self.sample_count, self.sample_count + data.shape[0]))
          self.sample_count += data.shape[0]
  def __len__(self):
    return self.sample_count
  def __getitem__(self, idx):
    which_file = list(filtered(lambda x: x[1] <= idx < x[2], self.file_list))
    npy_path, lower, upper = which_file[0]
    sample = np.load(npy_path)[idx - lower]
    potential = sample[3]
    density = np.reshape(sample[4:4+9**3], (9,9,9))
    grad_x = np.reshape(sample[4+9**3:4+(9**3)*2], (9,9,9))
    grad_y = np.reshape(sample[4+(9**3)*2:4+(9**3)*3], (9,9,9))
    grad_z = np.reshape(sample[4+(9**3)*3:4+(9**3)*4], (9,9,9))
    sample = {'rho': np.stack([density, grad_x, grad_y, grad_z], axis = -1), 'potential': potential}

if __name__ == "__main__":
  pass
