#!/usr/bin/python3

import math
import torch
from torch import nn
import torch.nn.functional as F

class SwitchGate(nn.Module):

    def __init__(
        self,
        dim,
        num_experts: int,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, use_aux_loss=False):

        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)

        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            load = gate_scores.sum(0)  # Sum over all examples
            importance = gate_scores.sum(1)  # Sum over all experts

            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None

class SwitchMoE(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, self.dim * self.mult, bias = True),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.dim * self.mult, self.dim, bias = True))
                for _ in range(num_experts)
            ]
        )

        self.gate = SwitchGate(
            dim,
            num_experts,
            capacity_factor,
        )

    def forward(self, x: torch.Tensor):

        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )

        # Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )

        return moe_output, loss

class MLPMixer(nn.Module):
  def __init__(self, **kwargs):
    super(MLPMixer, self).__init__()
    self.hidden_dim = kwargs.get('hidden_dim', 768)
    self.num_blocks = kwargs.get('num_blocks', 12)
    self.tokens_mlp_dim = kwargs.get('tokens_mlp_dim', 384)
    self.channels_mlp_dim = kwargs.get('channels_mlp_dim', 3072)
    self.drop_rate = kwargs.get('drop_rate', 0.1)

    self.layernorm1 = nn.LayerNorm((9**3, 4))
    self.dense = nn.Linear(4, self.hidden_dim)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(self.drop_rate)
    layers = dict()
    for i in range(self.num_blocks):
      layers.update({
        'layernorm1_%d' % i: nn.LayerNorm((self.hidden_dim, 9**3)),
        'linear1_%d' % i: nn.Linear(9**3, self.tokens_mlp_dim),
        'gelu1_%d' % i: nn.GELU(),
        'linear2_%d' % i: nn.Linear(self.tokens_mlp_dim, 9**3),
        'ffn_%d' % i: SwitchMoE(self.hidden_dim, self.channels_mlp_dim, self.hidden_dim, 3),
      })
    self.layers = nn.ModuleDict(layers)
    self.layernorm2 = nn.LayerNorm((9**3,self.hidden_dim))
  def forward(self, inputs):
    # inputs.shape = (batch, 4, 9, 9, 9)
    results = torch.flatten(inputs, 2) # results.shape = (batch, 4, 9**3)
    results = torch.permute(results, (0,2,1)) # results.shape = (batch, 9**3, 4)
    results = self.layernorm1(results)
    results = self.dense(results)
    results = self.gelu(results)
    results = self.dropout(results)

    for i in range(self.num_blocks):
      # 1) spatial mixing
      skip = results
      results = torch.permute(results, (0,2,1)) # results.shape = (batch, channel, 9**3)
      results = self.layers['layernorm1_%d' % i](results)
      results = self.layers['linear1_%d' % i](results)
      results = self.layers['gelu1_%d' % i](results)
      results = self.layers['linear2_%d' % i](results)
      results = torch.permute(results, (0,2,1)) # resutls.shape = (batch, 9**3, channel)
      results = results + skip
      # 2) channel mixing
      skip = results
      results, _ = self.layers['ffn_%d' % i](results)
      results = results + skip
    results = self.layernorm2(results) # results.shape = (batch, 9**3, channel)
    results = torch.mean(results, dim = 1) # results.shape = (batch, channel)
    return results

class Predictor(nn.Module):
  def __init__(self, **kwargs):
    super(Predictor, self).__init__()
    self.predictor = MLPMixer(**kwargs)
    self.dense1 = nn.Linear(kwargs.get('hidden_dim'), 1)
  def forward(self, inputs):
    results = self.predictor(inputs)
    results = self.dense1(results)
    return results

class PredictorSmall(nn.Module):
  def __init__(self):
    super(PredictorSmall, self).__init__()
    kwargs = {'hidden_dim': 256, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 256*4, 'drop_rate': 0.1}
    self.predictor = Predictor(**kwargs)
  def forward(self, inputs):
    return self.predictor(inputs)

class PredictorBase(nn.Module):
  def __init__(self):
    super(PredictorBase, self).__init__()
    kwargs = {'hidden_dim': 768, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072, 'drop_rate': 0.1}
    self.predictor = Predictor(**kwargs)
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
