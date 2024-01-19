#!/usr/bin/python3

import torch
from torch import nn

class Attention(nn.Module):
  def __init__(self, **kwargs):
    super(Attention, self).__init__(**kwargs)
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
    super(ABlock, self).__init__(**kwargs)
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

if __name__ == "__main__":
  att = Attention()
  inputs = torch.randn(2, 768, 10)
  results = att(inputs)
  print(results.shape)
  ablock = ABlock(input_size = 9)
  inputs = torch.randn(2, 768, 9, 9, 9)
  results = ablock(inputs)
  print(results.shape)
