"""Parameter-matched autoregressive baseline over the same pre-token stream.

Standard decoder-only LM sharing the transformer core with the NCM generator
(same n_embd/heads/mlp → param budget within ~10%). Decode uses a KV cache —
the honest real-world AR decode path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_swe.model.common import (Block, RMSNorm, causal_mask,
                                  precompute_rope)


class ARBaseline(nn.Module):
    def __init__(self, vocab_size: int, n_embd=896, n_layer=4, n_head_q=14,
                 n_head_kv=2, dropout=0.0, mlp_hidden=4864, glu=True,
                 use_qk_norm=False, rope_base=1_000_000.0, bias=True,
                 context_length=128):
        super().__init__()
        self.context_length = context_length
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(context_length, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head_q, n_head_kv, dropout, mlp_hidden,
                  use_qk_norm=use_qk_norm, rope=True, glu=glu, bias=bias)
            for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied
        hd = n_embd // n_head_q
        cos, sin = precompute_rope(hd, context_length + 8, base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.register_buffer("mask", causal_mask(context_length).reshape(
            1, 1, context_length, context_length), persistent=False)

    def forward(self, ids, targets=None, past=None):
        B, T = ids.shape
        off = past[0][0].shape[2] if past is not None else 0
        pos = torch.arange(off, off + T, device=ids.device)
        x = self.wte(ids) + self.wpe(pos)
        cos = self.rope_cos[off:off + T]
        sin = self.rope_sin[off:off + T]
        mask = self.mask[:, :, :T, :T] if past is None else None
        new_past = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, mask, cos, sin,
                        past[i] if past is not None else None)
            new_past.append(kv)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, new_past
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               targets[:, 1:].reshape(-1))
        return loss, new_past

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=1.0,
                 top_k=0, top_p=1.0, eos_id=None):
        self.eval()
        ids = prompt_ids
        past = None
        for _ in range(max_new_tokens):
            logits, past = self(ids if past is None else ids[:, -1:],
                                past=past)
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]), -1).values[:, -1:]
                logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sl, -1)
                sl = sl.masked_fill((probs.cumsum(-1) - probs) > top_p, -1e9)
                logits = torch.full_like(logits, -1e9).scatter(-1, si, sl)
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return ids
