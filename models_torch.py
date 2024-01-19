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
    results = self.dense1(inputs) # results.shape = (batch, 3 * channel, seq_len)
    b, _, s = results.shape
    results = results.view(b, 3, self.channel // self.num_heads, self.num_heads, s) # results.shape = (batch, 3, channel // head, head, seq_len)
    results = torch.permute(results, (0, 1, 3, 4, 2)) # results.shape = (batch, 3, head, seq_len, channel // head)
    q, k, v = results[:,0,...], results[:,1,...], results[:,2,...] # shape = (batch, head, seq_len, channel // head)
    qk = torch.matmul(q, torch.transpose(k, 2, 3)) # qk.shape = (batch, head, seq_len, seq_len)
    attn = torch.softmax(qk, dim = -1) # attn.shape = (batch, head, seq_len, seq_len)
    attn = self.dropout1(attn)
    qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3)) # qkv.shape = (batch, seq_len, head, channel // head)
    qkv = torch.transpose(qkv.view(b, s, self.channel), 1, 2) # qkv.shape = (batch, channel, seq_len)
    results = self.dense2(qvk) # results.shape = (batch, channel, seq_len)
    results = self.dropout2(results)
    return results

if __name__ == "__main__":
  att = Attention()
  inputs = torch.randn(2, 768, 10)
  results = att(inputs)
  print(results)
