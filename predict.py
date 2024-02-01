#!/usr/bin/python3

from os.path import join
import numpy as np
import torch
from torch import load, device
from models_torch import PredictorSmall

class Predict(object):
  def __init__(self, ckpt_path):
    ckpt = load(join(ckpt_path, "model.pth"))
    self.model = PredictorSmall(in_channel = 4, groups = 1).to(torch.device('cuda'))
    self.model.load_state_dict(ckpt['state_dict'])
    self.model.eval()
  def predict(self, inputs):
    # NOTE: inputs.shape = (batch, 4, 9, 9, 9)
    if type(inputs) is np.ndarray:
      inputs = torch.from_numpy(inputs.astype(np.float32))
    inputs = inputs.to(torch.device('cuda'))
    return np.log(self.model(inputs).cpu().detach().numpy())

