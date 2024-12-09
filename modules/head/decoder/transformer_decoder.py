import imp
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class LayerNorm(nn.Module):
    r"""
    Layer normalization.
    """

    def __init__(self, hidden_dims, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dims))
        self.bias = nn.Parameter(torch.zeros(hidden_dims))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MLP(nn.Module):
    def __init__(self, hidden_dims, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_dims
        self.linear = nn.Linear(hidden_dims, out_features)
        self.layer_norm = LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states


class SelfAttention(nn.Module):
    def __init__(self, hidden_dims, num_attn_heads=1):
        super(SelfAttention, self).__init__()
        self.num_attn_heads = num_attn_heads
        self.attn_head_dims = hidden_dims // num_attn_heads
        self.num_qkv = 1
        self.query = nn.Linear(hidden_dims, hidden_dims * self.num_qkv)
        self.key = nn.Linear(hidden_dims, hidden_dims * self.num_qkv)
        self.value = nn.Linear(hidden_dims, hidden_dims * self.num_qkv)
        self.softmax = nn.Softmax(dim=-1)

    def get_extended_attention_mask(self, attention_mask):
        """
        1 in attention_mask stands for doing attention, 0 for not doing attention.

        After this function, 1 turns to 0, 0 turns to -10000.0

        Because the -10000.0 will be fed into softmax and -10000.0 can be thought as 0 in softmax.
        """
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1, self.num_attn_heads, self.attn_head_dims)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states:Tensor, attention_mask:Tensor=None) -> Tensor:
        bs = hidden_states.shape[0]
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query / math.sqrt(self.attn_head_dims), key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = self.softmax(attention_scores)
        out = torch.matmul(attention_probs, value)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(bs, -1, self.num_attn_heads * self.attn_head_dims)
        return out


class CrossAttention(SelfAttention):
    def __init__(self, hidden_dims, num_attn_heads=1):
        super(CrossAttention, self).__init__(hidden_dims, num_attn_heads)

    def forward(self, hs_query, hs_key, attention_mask=None):
        bs = hs_query.shape[0]
        query = self.query(hs_query)
        key = self.key(hs_key)
        value = self.value(hs_key)

        query = self.transpose_for_scores(query) 
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        attention_scores = torch.matmul(query / math.sqrt(self.attn_head_dims), key.transpose(-1, -2))
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)
        attention_probs = self.softmax(attention_scores)
        out = torch.matmul(attention_probs, value)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(bs, -1, self.num_attn_heads * self.attn_head_dims)
        return out


class AgentSelfAttention(nn.Module):
    def __init__(self, hidden_dims, depth=2):
        super(AgentSelfAttention, self).__init__()
        self.layers = nn.ModuleList([SelfAttention(hidden_dims, num_attn_heads=8) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_dims) for _ in range(depth)])

    def forward(self, hidden_states, attention_mask=None):        
        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)       # self-attn
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp                       # 残差连接
            hidden_states = self.layers_2[layer_index](hidden_states)  # Layer Norm
        return hidden_states