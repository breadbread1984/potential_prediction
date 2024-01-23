#!/usr/bin/python3

import math
import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
  def __init__(self, **kwargs):
    super(Attention, self).__init__()
    self.channel = kwargs.get('channel', 768)
    self.num_heads = kwargs.get('num_heads', 8)
    self.qkv_bias = kwargs.get('qkv_bias', False)
    self.drop_rate = kwargs.get('drop_rate', 0.1)

    self.dense1 = nn.Linear(self.channel, self.channel * 3, bias = self.qkv_bias)
    self.dense2 = nn.Linear(self.channel, self.channel, bias = self.qkv_bias)
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.dropout2 = nn.Dropout(self.drop_rate)
  def forward(self, inputs):
    # inputs.shape = (batch, channel, seq_len)
    results = self.dense1(torch.transpose(inputs, 1, 2)) # results.shape = (batch, seq_len, 3 * channel)
    b, s, _ = results.shape
    results = torch.reshape(results, (b, s, 3, self.num_heads, self.channel // self.num_heads)) # results.shape = (batch, seq_len, 3, head, channel // head)
    results = torch.permute(results, (0, 2, 3, 1, 4)) # results.shape = (batch, 3, head, seq_len, channel // head)
    q, k, v = results[:,0,...], results[:,1,...], results[:,2,...] # shape = (batch, head, seq_len, channel // head)
    qk = torch.matmul(q, torch.transpose(k, 2, 3)) # qk.shape = (batch, head, seq_len, seq_len)
    attn = torch.softmax(qk, dim = -1) # attn.shape = (batch, head, seq_len, seq_len)
    attn = self.dropout1(attn)
    qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3)) # qkv.shape = (batch, seq_len, head, channel // head)
    qkv = torch.reshape(qkv, (b, s, self.channel)) # qkv.shape = (batch, seq_len, channel)
    results = self.dense2(qkv) # results.shape = (batch, seq_len, channel)
    results = self.dropout2(results)
    results = torch.transpose(results, 1, 2) # results.shape = (batch, channel, seq_len)
    return results

class ABlock(nn.Module):
  def __init__(self, input_size, **kwargs):
    super(ABlock, self).__init__()
    self.input_size = input_size
    self.channel = kwargs.get('channel', 768)
    self.mlp_ratio = kwargs.get('mlp_ratio', 4)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
    self.num_heads = kwargs.get('num_heads', 8)
    self.qkv_bias = kwargs.get('qkv_bias', False)
    self.groups = kwargs.get('groups', 1)

    self.conv1 = nn.Conv3d(self.channel, self.channel, kernel_size = (3,3,3), padding = 'same', groups = self.groups)
    self.conv2 = nn.Conv3d(self.channel, self.channel * self.mlp_ratio, kernel_size = (1,1,1), padding = 'same', groups = self.groups)
    self.conv3 = nn.Conv3d(self.channel * self.mlp_ratio, self.channel, kernel_size = (1,1,1), padding = 'same', groups = self.groups)
    self.gelu = nn.GELU()
    self.layernorm1 = nn.LayerNorm([self.channel, self.input_size, self.input_size, self.input_size])
    self.layernorm2 = nn.LayerNorm([self.channel, self.input_size, self.input_size, self.input_size])
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.dropout2 = nn.Dropout(self.drop_rate)
    self.atten = Attention(**kwargs)
  def forward(self, inputs):
    # inputs.shape = (batch, c, t, h, w)
    # positional embedding
    skip = inputs
    pos_embed = self.conv1(inputs) # pos_embed.shape = (batch, channel, t, h, w)
    results = skip + pos_embed
    # attention
    skip = results
    results = self.layernorm1(results)
    b, c, t, h, w = results.shape
    results = torch.reshape(results, (b, c, t * h * w)) # results.shape = (batch, channel, t * h * w)
    results = self.atten(results) # results.shape = (batch, channel, t * h * w)
    results = torch.reshape(results, (b, c, t, h, w)) # results.shape = (batch, channel, t, h, w)
    results = skip + results
    # mlp
    skip = results
    results = self.layernorm2(results)
    results = self.conv2(results) # results.shape = (batch, channel * mlp_ratio, t, h, w)
    results = self.gelu(results)
    results = self.dropout1(results)
    results = self.conv3(results) # results.shape = (batch, channel, t, h, w)
    results = self.dropout2(results)
    results = skip + results
    return results

class Conv3dSame(nn.Conv3d):
  def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    it, ih, iw = x.size()[-3:]
    pad_t = self.calc_same_pad(i = it, k = self.kernel_size[0], s = self.stride[0], d = self.dilation[0])
    pad_h = self.calc_same_pad(i = ih, k = self.kernel_size[1], s = self.stride[1], d = self.dilation[1])
    pad_w = self.calc_same_pad(i = iw, k = self.kernel_size[2], s = self.stride[2], d = self.dilation[2])
    if pad_t > 0 or pad_h > 0 or pad_w > 0:
      x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_t // 2, pad_t - pad_t // 2])
    return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Extractor(nn.Module):
  def __init__(self, **kwargs):
    super(Extractor, self).__init__()
    self.in_channel = kwargs.get('in_channel', 3)
    self.out_channel = kwargs.get('out_channel', None)
    self.hidden_channels = kwargs.get('hidden_channels', [128, 512])
    self.depth = kwargs.get('depth', [8, 3])
    self.mlp_ratio = kwargs.get('mlp_ratio', 4.)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
    self.qkv_bias = kwargs.get('qkv_bias', False)
    self.num_heads = kwargs.get('num_heads', 8)
    self.groups = kwargs.get('groups', 1)
    
    #self.batchnorm1 = nn.BatchNorm3d(4)
    self.batchnorm2 = nn.BatchNorm3d(self.hidden_channels[1])
    self.conv1 = nn.Conv3d(4, self.hidden_channels[0], kernel_size = (3,3,3), padding = 'same')
    self.conv2 = Conv3dSame(self.hidden_channels[0], self.hidden_channels[1], kernel_size = (3,3,3), stride = 3, groups = self.groups)
    self.conv3 = Conv3dSame(self.hidden_channels[1], self.hidden_channels[1], kernel_size = (3,3,3), stride = 3, groups = self.groups)
    self.layernorm1 = nn.LayerNorm([self.hidden_channels[0], 9, 9, 9])
    self.layernorm2 = nn.LayerNorm([self.hidden_channels[1], 3, 3, 3])
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.block1 = nn.ModuleList([ABlock(input_size = 9, channel = self.hidden_channels[0], qkv_bias = self.qkv_bias, num_heads = self.num_heads, **kwargs) for i in range(self.depth[0])])
    self.block2 = nn.ModuleList([ABlock(input_size = 3, channel = self.hidden_channels[1], qkv_bias = self.qkv_bias, num_heads = self.num_heads, **kwargs) for i in range(self.depth[1])])
  def forward(self, inputs):
    # inputs.shape = (batch, 4, 9, 9, 9)
    #results = self.batchnorm1(inputs)
    results = self.conv1(inputs) # results.shape = (batch, hidden_channels[0], 9, 9, 9)
    results = self.layernorm1(results)
    results = self.dropout1(results)
    # do attention only when the feature shape is small enough
    for i in range(self.depth[0]):
      results = self.block1[i](results)
    results = self.conv2(results) # results.shape = (batch, hidden_channels[1], 3, 3, 3)
    results = self.layernorm2(results)
    for i in range(self.depth[1]):
      results = self.block2[i](results)
    results = self.conv3(results) # results.shape = (batch, hidden_channels[1], 1, 1, 1)
    results = self.batchnorm2(results)
    results = torch.squeeze(results, (2,3,4)) # results.shape = (batch, hidden_channels[1])
    return results

class Predictor(nn.Module):
  def __init__(self, **kwargs):
    super(Predictor, self).__init__()
    self.predictor = Extractor(**kwargs)
    self.dense1 = nn.Linear(kwargs.get('hidden_channels')[-1], 20)
    self.gelu = nn.GELU()
    self.dense2 = nn.Linear(20, 1)
    self.tanh = nn.Tanh()
  def forward(self, inputs):
    results = self.predictor(inputs)
    results = self.dense1(results)
    results = self.gelu(results)
    results = self.dense2(results)
    results = self.tanh(results)
    return results

class PredictorSmall(nn.Module):
  def __init__(self, **kwargs):
    super(PredictorSmall, self).__init__()
    hidden_channels = kwargs.get('hidden_channels', [128, 512])
    depth = kwargs.get('depth', [1, 1])
    self.predictor = Predictor(hidden_channels = hidden_channels, depth = depth, **kwargs)
  def forward(self, inputs):
    return self.predictor(inputs)

class PredictorBase(nn.Module):
  def __init__(self, **kwargs):
    super(PredictorBase, self).__init__()
    hidden_channels = kwargs.get('hidden_channels', [128, 512])
    depth = kwargs.get('depth', [20, 7])
    self.predictor = Predictor(hidden_channels = hidden_channels, depth = depth, **kwargs)
  def forward(self, inputs):
    return self.predictor(inputs)

if __name__ == "__main__":
  att = Attention()
  inputs = torch.randn(2, 768, 10)
  results = att(inputs)
  print(results.shape)
  ablock = ABlock(input_size = 9)
  inputs = torch.randn(2, 768, 9, 9, 9)
  results = ablock(inputs)
  print(results.shape)
  predictor = PredictorSmall(in_channel = 4, out_channel = 768, groups = 1)
  inputs = torch.randn(2, 4, 9, 9, 9)
  results = predictor(inputs)
  print(results.shape)
