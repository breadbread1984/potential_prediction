#!/usr/bin/python3

import math
import torch
from torch import nn
from mixture_of_experts import MoE

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

class SwiGLU(nn.Module):
  def __init__(self, **kwargs):
    super(SwiGLU, self).__init__()
    self.channel = kwargs.get('channel', 768)
    self.intermediate = math.floor(self.channel * 4 * 2 / 3)
    self.gate_proj = nn.Linear(self.channel, self.intermediate, bias = False)
    self.up_proj = nn.Linear(self.channel, self.intermediate, bias = False)
    self.down_proj = nn.Linear(self.intermediate, self.channel, bias = False)
    self.act_fn = nn.SiLU()
  def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

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
    self.num_experts = kwargs.get('num_experts', 3)

    self.layernorm1 = nn.LayerNorm([self.channel, self.input_size, self.input_size, self.input_size])
    self.dropout0 = nn.Dropout(self.drop_rate)
    self.atten = Attention(**kwargs)
    self.moe = MoE(dim = self.channel, num_experts = self.num_experts, experts = SwiGLU(**kwargs))
  def forward(self, inputs):
    # inputs.shape = (batch, c, t, h, w)
    # attention
    skip = inputs
    results = self.layernorm1(inputs)
    b, c, t, h, w = results.shape
    results = torch.reshape(results, (b, c, t * h * w)) # results.shape = (batch, channel, t * h * w)
    results = self.atten(results) # results.shape = (batch, channel, t * h * w)
    results = torch.reshape(results, (b, c, t, h, w)) # results.shape = (batch, channel, t, h, w)
    results = self.dropout0(results)
    results = skip + results
    # mlp
    skip = results
    results = torch.flatten(results, start_dim = 2) # results.shape = (batch, channel, 9**3)
    results = torch.permute(results, (0,2,1)) # results.shape = (batch, 9**3, channel)
    results, _ = self.moe(results)
    results = torch.permute(results, (0,2,1)) # results.shape = (batch, channel, 9**3)
    b, c, _ = results.shape
    results = torch.reshape(results, (b, c, 9, 9, 9)) # results.shape = (batch, channel, 9, 9, 9)
    results = skip + results
    return results

class Extractor(nn.Module):
  def __init__(self, **kwargs):
    super(Extractor, self).__init__()
    self.in_channel = kwargs.get('in_channel', 3)
    self.hidden_channels = kwargs.get('hidden_channels', 512)
    self.depth = kwargs.get('depth', 12)
    self.mlp_ratio = kwargs.get('mlp_ratio', 4.)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
    self.qkv_bias = kwargs.get('qkv_bias', False)
    self.num_heads = kwargs.get('num_heads', 8)
    self.groups = kwargs.get('groups', 1)
    
    self.gelu = nn.GELU()
    self.tanh = nn.Tanh()
    self.conv1 = nn.Conv3d(4, self.hidden_channels, kernel_size = (1,1,1), padding = 'same')
    self.conv2 = nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size = (1,1,1), padding = 'same', groups = self.groups)
    self.layernorm1 = nn.LayerNorm([4, 9, 9, 9])
    self.layernorm2 = nn.LayerNorm([self.hidden_channels, 9, 9, 9])
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.blocks = nn.ModuleList([ABlock(input_size = 9, channel = self.hidden_channels, qkv_bias = self.qkv_bias, num_heads = self.num_heads, **kwargs) for i in range(self.depth)])
  def forward(self, inputs):
    # inputs.shape = (batch, 4, 9, 9, 9)
    results = self.layernorm1(inputs)
    results = self.conv1(results) # results.shape = (batch, hidden_channels[0], 9, 9, 9)
    results = self.gelu(results)
    results = self.dropout1(results)
    # do attention only when the feature shape is small enough
    for i in range(self.depth):
      results = self.blocks[i](results)
    results = self.layernorm2(results)
    results = self.conv2(results) # results.shape = (batch, hidden_channels, 9, 9, 9)
    results = self.tanh(results) # results.shape = (batch, hidden_channels, 9, 9, 9)
    #results = torch.mean(results, dim = (2,3,4)) # results.shape = (batch, hidden_channels)
    return results

class Predictor(nn.Module):
  def __init__(self, **kwargs):
    super(Predictor, self).__init__()
    self.predictor = Extractor(**kwargs)
    self.dense1 = nn.Linear(kwargs.get('hidden_channels'), 1)
  def forward(self, inputs):
    results = self.predictor(inputs)
    results = self.dense1(results[:,:,0,0,0])
    return results

class PredictorSmall(nn.Module):
  def __init__(self, **kwargs):
    super(PredictorSmall, self).__init__()
    hidden_channels = kwargs.get('hidden_channels', 512)
    depth = kwargs.get('depth', 12)
    self.predictor = Predictor(hidden_channels = hidden_channels, depth = depth, **kwargs)
  def forward(self, inputs):
    return self.predictor(inputs)

class PredictorBase(nn.Module):
  def __init__(self, **kwargs):
    super(PredictorBase, self).__init__()
    hidden_channels = kwargs.get('hidden_channels', 512)
    depth = kwargs.get('depth', 24)
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
  predictor = PredictorSmall(in_channel = 4, groups = 1)
  inputs = torch.randn(2, 4, 9, 9, 9)
  results = predictor(inputs)
  print(results.shape)
