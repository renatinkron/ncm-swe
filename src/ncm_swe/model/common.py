"""Shared transformer core used by the NCM generator and the AR baseline.

Pre-LN blocks, RoPE, optional QK-norm, GQA (separate q/kv head counts),
SwiGLU or GELU MLP — dimension-compatible with Qwen2.5-Coder-0.5B so its
weights can transplant 1:1 for the warm-start arm.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        return (x * x.pow(2).mean(-1, keepdim=True).add_(self.eps).rsqrt_()
                ).type_as(x) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-6)


def precompute_rope(head_dim: int, max_t: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_t).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); cos/sin: (T, D/2) — rotary on first D/2 pairs (Qwen style:
    # rotate full dim split in half)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head_q: int, n_head_kv: int,
                 dropout: float, use_qk_norm: bool = True, rope: bool = True,
                 bias: bool = True):
        super().__init__()
        assert n_embd % n_head_q == 0
        self.n_head_q = n_head_q
        self.n_head_kv = n_head_kv
        self.head_dim = n_embd // n_head_q
        self.rope = rope
        self.c_attn = nn.Linear(n_embd,
                                n_embd + 2 * n_head_kv * self.head_dim,
                                bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

    def qkv(self, x):
        B, T, _ = x.shape
        q, k, v = self.c_attn(x).split(
            [self.n_head_q * self.head_dim,
             self.n_head_kv * self.head_dim,
             self.n_head_kv * self.head_dim], dim=2)
        q = q.view(B, T, self.n_head_q, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def attend(self, q, k, v, attn_mask=None, cos=None, sin=None,
               past_kv=None):
        if self.rope and cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v)  # pre-repeat (GQA-compact) cache
        if self.n_head_kv != self.n_head_q:
            rep = self.n_head_q // self.n_head_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0)
        return y, new_kv

    def forward(self, x, attn_mask=None, cos=None, sin=None, past_kv=None):
        q, k, v = self.qkv(x)
        y, kv = self.attend(q, k, v, attn_mask, cos, sin, past_kv)
        B, T = x.shape[0], x.shape[1]
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.drop(self.c_proj(y)), kv


class MLP(nn.Module):
    def __init__(self, n_embd: int, hidden: int, dropout: float,
                 glu: bool = False, bias: bool = True):
        super().__init__()
        self.glu = glu
        if glu:
            self.gate = nn.Linear(n_embd, hidden, bias=bias)
            self.up = nn.Linear(n_embd, hidden, bias=bias)
        else:
            self.c_fc = nn.Linear(n_embd, hidden, bias=bias)
        self.c_proj = nn.Linear(hidden, n_embd, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.glu:
            x = F.silu(self.gate(x)) * self.up(x)
        else:
            x = F.gelu(self.c_fc(x))
        return self.drop(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head_q: int, n_head_kv: int,
                 dropout: float, mlp_hidden: int, use_qk_norm: bool = True,
                 rope: bool = True, glu: bool = False, bias: bool = True):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head_q, n_head_kv, dropout,
                                  use_qk_norm, rope, bias)
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, mlp_hidden, dropout, glu, bias)

    def forward(self, x, attn_mask=None, cos=None, sin=None, past_kv=None):
        a, kv = self.attn(self.ln_1(x), attn_mask, cos, sin, past_kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, kv


def causal_mask(T: int, device=None) -> torch.Tensor:
    return torch.triu(torch.full((T, T), float("-inf"), device=device), 1)


def level_id_tensor(levels) -> torch.Tensor:
    return torch.cat([torch.full((l,), i, dtype=torch.long)
                      for i, l in enumerate(levels)])
