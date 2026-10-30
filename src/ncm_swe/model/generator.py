"""NCM stage-2: level-parallel generator.

Semantics follow the upstream NextConceptTransformer (block_diagonal pattern):
positions attend within their own level block (bidirectional, parallel
prediction) plus the level-0 query position. Cross-level conditioning is baked
into the input embeddings: the input at each position of level k is the VQ
accumulated quantized embedding z_hat of levels < k interpolated to L_k
(`MultiscaleResidualVectorQuantizer.idx_to_ncm_input`).

Sequence layout: [q0] + chunks for levels 1..K-1
  q0            — learned query position; predicts level-0 codes.
  chunk k       — L_k positions; input = proj(z_hat_{<k}); target = level-k codes.

Generation: K sequential passes (one per level), all codes of a level sampled
in parallel — the NCM speedup mechanism vs L sequential AR steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_swe.model.common import (Block, RMSNorm, causal_mask,
                                  level_id_tensor, precompute_rope)


class LevelGenerator(nn.Module):
    def __init__(self, vqvae, n_embd=896, n_layer=4, n_head_q=14,
                 n_head_kv=2, dropout=0.0, mlp_hidden=4864, glu=True,
                 use_qk_norm=False, rope_base=1_000_000.0, bias=True,
                 shared_head=True):
        super().__init__()
        self.vqvae = vqvae  # frozen tokenizer
        self.levels = list(vqvae.vq.levels)          # e.g. [1,8,...,128]
        self.chunk_levels = self.levels[1:]           # predicted in main seq
        self.vocab_sizes = list(vqvae.vq.vqvae_vocab_sizes)
        self.V = max(self.vocab_sizes)
        self.embed_dim = vqvae.vq.embed_dim
        self.T = sum(self.chunk_levels)               # main seq positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.head_dim = n_embd // n_head_q

        self.q0 = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.proj_in = nn.Linear(self.embed_dim, n_embd, bias=False)
        self.pos = nn.Embedding(1 + self.T, n_embd)
        self.lvl = nn.Embedding(len(self.levels), n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head_q, n_head_kv, dropout, mlp_hidden,
                  use_qk_norm=use_qk_norm, rope=True, glu=glu, bias=bias)
            for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        if shared_head:
            self.head = nn.Linear(n_embd, self.V, bias=False)
        else:
            self.head = nn.ModuleList(
                [nn.Linear(n_embd, v, bias=False) for v in self.vocab_sizes])
        self.lvl_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(v)) for v in self.vocab_sizes])
        self.shared_head = shared_head

        cos, sin = precompute_rope(self.head_dim, 1 + self.T,
                                   base=rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # mask: pos0 (q0) sees only itself; main pos sees q0 + own level block
        T = self.T
        lvl_ids = level_id_tensor(self.chunk_levels)  # 0..K-2 for main pos
        m = torch.full((T + 1, T + 1), float("-inf"))
        m[0, 0] = 0.0
        m[1:, 0] = 0.0  # targets attend to q0
        same = lvl_ids[:, None] == lvl_ids[None, :]
        m[1:, 1:] = torch.where(same, 0.0, float("-inf"))
        self.register_buffer("attn_mask", m.reshape(1, 1, T + 1, T + 1),
                             persistent=False)
        self.register_buffer("lvl_ids", lvl_ids, persistent=False)
        self.apply(self._init_weights)
        nn.init.normal_(self.q0, std=0.02)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _head_logits(self, h, level_idx):
        if self.shared_head:
            z = self.head(h)[:, :, : self.vocab_sizes[level_idx]]
        else:
            z = self.head[level_idx](h)
        return z + self.lvl_bias[level_idx]

    def forward(self, idx):
        """idx: (B, sum(levels)). Returns (logits_l0 (B,l0,V0), logits_main
        (B,T,V_padded_per_pos->use per-level CE helper)) for training."""
        B = idx.shape[0]
        with torch.no_grad():
            x = self.vqvae.vq.idx_to_ncm_input(idx)  # (B,T,D_vq)
        h0 = self.q0.expand(B, -1, -1) + self.pos.weight[0] + self.lvl.weight[0]
        hmain = self.proj_in(x) + self.pos.weight[1:] \
            + self.lvl.weight[self.lvl_ids + 1]
        h = torch.cat([h0, hmain], dim=1)
        for blk in self.blocks:
            h, _ = blk(h, self.attn_mask[:, :, : h.shape[1], : h.shape[1]],
                       self.rope_cos[: h.shape[1]], self.rope_sin[: h.shape[1]])
        h = self.ln_f(h)
        logits_l0 = self._head_logits(h[:, :1], 0)
        # per-position head with per-level slicing -> gather into padded tensor
        outs = []
        ofs = 1
        for i, l in enumerate(self.chunk_levels):
            outs.append(self._head_logits(h[:, ofs:ofs + l], i + 1))
            ofs += l
        return logits_l0, outs  # outs[i]: (B, L_{i+1}, V_{i+1})

    def loss(self, idx):
        logits_l0, outs = self.forward(idx)
        l0 = self.levels[0]
        t0 = idx[:, :l0]
        loss_l0 = F.cross_entropy(
            logits_l0.reshape(-1, logits_l0.shape[-1]), t0.reshape(-1))
        total = loss_l0
        ofs = l0
        per_level = {"l0": float(loss_l0.detach())}
        for i, out in enumerate(outs):
            t = idx[:, ofs:ofs + self.chunk_levels[i]]
            ce = F.cross_entropy(out.reshape(-1, out.shape[-1]), t.reshape(-1))
            per_level[f"l{i+1}"] = round(float(ce.detach()), 4)
            total = total + ce
            ofs += self.chunk_levels[i]
        return total / (len(outs) + 1), per_level

    # ----------------------------- sampling ---------------------------------

    def _chunks_from_codes(self, codes_by_level: list[torch.Tensor], upto: int):
        """Build input chunks to predict levels 1..upto from codes 0..upto-1.

        Mirrors vq.idx_to_ncm_input on partial codes.
        """
        vq = self.vqvae.vq
        L_full = self.levels[-1]
        z_hat = torch.zeros(codes_by_level[0].shape[0], L_full,
                            self.embed_dim, device=codes_by_level[0].device)
        chunks = []
        for j in range(upto):
            emb_j = vq.embeddings[j]
            z_q = emb_j(codes_by_level[j])
            if j != len(self.levels) - 1:
                z_q = vq._up_to_full(z_q, j, L_full)
            z_q = vq._apply_quant_resi(z_q, j)
            z_hat = z_hat + z_q
            if j + 1 < len(self.levels):
                chunks.append(F.interpolate(
                    z_hat.transpose(-1, -2), size=(self.levels[j + 1],),
                    mode="area").transpose(-1, -2))
        return chunks  # chunks[k] predicts level k+1

    def _forward_partial(self, codes_by_level):
        """Run the stack over [q0] + chunks predicting levels 1..len(codes)."""
        B = codes_by_level[0].shape[0]
        k_next = len(codes_by_level)  # predict level index k_next
        chunks = self._chunks_from_codes(codes_by_level, k_next)
        h0 = self.q0.expand(B, -1, -1) + self.pos.weight[0] + self.lvl.weight[0]
        parts = [h0]
        pos = 1
        for i, ch in enumerate(chunks):
            parts.append(self.proj_in(ch) + self.pos.weight[pos:pos + ch.shape[1]]
                         + self.lvl.weight[i + 1])
            pos += ch.shape[1]
        h = torch.cat(parts, dim=1)
        T = h.shape[1]
        mask = self.attn_mask[:, :, :T, :T]
        for blk in self.blocks:
            h, _ = blk(h, mask, self.rope_cos[:T], self.rope_sin[:T])
        return self.ln_f(h)

    @staticmethod
    def _sample(logits, temperature=1.0, top_k=0, top_p=1.0):
        logits = logits / max(temperature, 1e-6)
        if top_k and top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.shape[-1]), -1).values[:, -1:]
            logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)
        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sl, -1)
            cut = (probs.cumsum(-1) - probs) > top_p
            sl = sl.masked_fill(cut, -1e9)
            logits = torch.full_like(logits, -1e9).scatter(-1, si, sl)
        return torch.multinomial(F.softmax(logits, -1), 1)

    @torch.no_grad()
    def generate(self, batch_size=1, temperature=1.0, top_k=0, top_p=1.0,
                 return_per_level=False):
        """Coarse→fine generation in K passes. Returns idx (B,sum(levels))."""
        self.eval()
        codes = []
        # level 0: q0 only
        h = self.q0.expand(batch_size, -1, -1) + self.pos.weight[0] \
            + self.lvl.weight[0]
        mask0 = torch.zeros(1, 1, 1, 1)
        for blk in self.blocks:
            h, _ = blk(h, mask0, self.rope_cos[:1], self.rope_sin[:1])
        h = self.ln_f(h)
        l0_logits = self._head_logits(h, 0).squeeze(1)
        codes.append(self._sample(l0_logits, temperature, top_k, top_p))
        # levels 1..K-1
        for k in range(1, len(self.levels)):
            h = self._forward_partial(codes)
            # last chunk = last L_k positions
            Lk = self.levels[k]
            logits = self._head_logits(h[:, -Lk:], k)
            codes.append(self._sample(
                logits.reshape(batch_size * Lk, -1), temperature, top_k,
                top_p).reshape(batch_size, Lk))
        idx = torch.cat(codes, dim=1)
        if return_per_level:
            return idx, codes
        return idx

    @torch.no_grad()
    def decode_text(self, idx, bpe):
        logits = self.vqvae.decode(idx.to(next(self.parameters()).device))
        ids = logits.argmax(-1)
        return [bpe.decode(row.tolist(), skip_special_tokens=True)
                for row in ids]
