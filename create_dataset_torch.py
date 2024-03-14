#!/usr/bin/python3

from os.path import join
import numpy as np
from torch.utils.data import Dataset

class RhoDataset(Dataset):
  def __init__(self, dataset_dir, divide = 'train'):
    assert divide in {'train', 'eval'}
    super(RhoDataset, self).__init__()
    file_name = {'train': 'alldata.npy', 'eval': 'rho_all.npy'}[divide]
    self.samples = np.load(join(dataset_dir, file_name))
    self.labels = np.load(join(dataset_dir, 'vxc_all.npy'))
    assert self.samples.shape[0] == self.labels.shape[0]
  def __len__(self):
    return self.samples.shape[0]
  def __getitem__(self, idx):
    x = np.reshape(self.samples[idx,3:], (1,11,11,11)).astype(np.float32) # x.shape = (1,11,11,11)
    y = np.expand_dims(self.labels[idx], axis = -1).astype(np.float32) # y.shape = (1)
    label = np.arcsinh(y)
    return x, label

