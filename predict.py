#!/usr/bin/python3

from os.path import join
import numpy as np
import torch
from torch import load, device
from models_torch import PredictorSmall

class Predict(object):
  def __init__(self, ckpt_path, postprocess = 'exp'):
    self.postprocess = postprocess
    ckpt = load(join(ckpt_path, "model.pth"))
    self.model = PredictorSmall().to(torch.device('cuda'))
    self.model.load_state_dict(ckpt['state_dict'])
    self.model.eval()
  def predict(self, inputs):
    # NOTE: inputs.shape = (batch, 4, 9, 9, 9)
    if type(inputs) is np.ndarray:
      inputs = torch.from_numpy(inputs.astype(np.float32))
    inputs = inputs.to(torch.device('cuda'))
    results = self.model(inputs).cpu().detach().numpy()
    if self.postprocess == 'exp':
      return np.log(results)
    elif self.postprocess == 'log':
      return -np.sqrt(np.exp(np.maximum(results,0.)) - 1)

