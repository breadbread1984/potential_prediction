#!/usr/bin/python3

from os.path import join
import numpy as np
from torch.utils.data import Dataset

class RhoDataset(Dataset):
  def __init__(self, npy_path):
    super(RhoDataset, self).__init__()
    self.samples = np.load(npy_path)
  def __len__(self):
    return self.samples.shape[0]
  def __getitem__(self, idx):
    x = np.reshape(self.samples[idx,:-1], (1,11**3+31)).astype(np.float32) # x.shape = (1,1362)
    y = self.samples[idx,-1:].astype(np.float32) # y.shape = (1)
    x = np.log10(x)
    return x, y

