"""Minimal multiscale residual quantizer — stable at CPU scale.

Duck-type replacement for the vendored MultiscaleResidualVectorQuantizer
(which exhibited codebook explosion at our scale: EMA division by
near-zero smoothed counts + learned residual amplification, yielding
single-code collapse and vq_loss ~1e11).

Differences vs upstream:
- Codebook learned ONLY via EMA of assigned inputs (true running mean —
  mathematically bounded by |z|, cannot explode).
- No learned quant_resi transform, no fine-drop/corruption aux.
- Commitment loss only (beta * mse(z_hat, z.detach-stopgrad on codebook)).

Same public interface: levels, embeddings, vqvae_vocab_sizes, embed_dim,
z_to_idx, idx_to_z_q, idx_to_ncm_input, forward(z)->(z_q, vq_loss, utils).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniMSRQ(nn.Module):
    def __init__(self, levels=(1, 8, 16, 32, 64, 128), vocab_size=2048,
                 embed_dim=192, decay=0.99, commitment_cost=0.25):
        super().__init__()
        self.levels = list(levels)
        self.vqvae_vocab_sizes = ([vocab_size] * len(levels)
                                 if isinstance(vocab_size, int)
                                 else list(vocab_size))
        self.embed_dim = embed_dim
        self.decay = decay
        self.commitment_cost = commitment_cost
        L_full = self.levels[-1]
        self.L_full = L_full

        embs = []
        for vs in self.vqvae_vocab_sizes:
            e = nn.Embedding(vs, embed_dim)
            nn.init.normal_(e.weight, std=0.1)
            e.weight.requires_grad_(False)   # EMA-only codebook
            embs.append(e)
        self.embeddings = nn.ModuleList(embs)
        # EMA accumulators (buffers — checkpointed)
        for i, vs in enumerate(self.vqvae_vocab_sizes):
            self.register_buffer(f"ema_n_{i}", torch.zeros(vs))
            self.register_buffer(f"ema_w_{i}",
                                 torch.zeros(vs, embed_dim))

    # ---- shared helpers --------------------------------------------------

    def _interp(self, x_BLC, size):
        return F.interpolate(x_BLC.transpose(-1, -2), size=size,
                             mode="linear", align_corners=False
                             ).transpose(-1, -2)

    def _up_to_full(self, h_BLC, level_idx, L=None):
        L = L or self.L_full
        if h_BLC.shape[1] == L:
            return h_BLC
        return self._interp(h_BLC, L)

    def _apply_quant_resi(self, h_BLC, level_idx):  # vendored API compat
        return h_BLC

    def _nearest(self, flat, emb_w):
        d = (flat.pow(2).sum(1, keepdim=True) + emb_w.pow(2).sum(1)
             - 2 * flat @ emb_w.t())
        return d.argmin(1)

    # ---- forward ---------------------------------------------------------

    def forward(self, z):
        z = z.float()
        B, L, C = z.shape
        z_rest = z.detach().clone()          # residual stream (no grad)
        z_hat = torch.zeros_like(z)
        vq_loss = 0.0
        per_util = []
        K = len(self.levels)

        for i, L_i in enumerate(self.levels):
            emb_i = self.embeddings[i]
            if i != K - 1:
                inp = self._interp(z_rest, L_i)      # downsample residual
            else:
                inp = z_rest
            flat = inp.reshape(-1, C).detach()
            if self.training and getattr(self, f"ema_n_{i}").sum() == 0:
                # cold start: seed codebook from this batch's inputs
                with torch.no_grad():
                    sel = torch.randint(0, flat.shape[0],
                                        (self.vqvae_vocab_sizes[i],),
                                        device=flat.device)
                    emb_i.weight.data.copy_(flat[sel])
            idx = self._nearest(flat, emb_i.weight)
            q = emb_i.weight[idx].view(B, L_i, C)

            if self.training:                        # EMA codebook update
                with torch.no_grad():
                    n = torch.zeros_like(getattr(self, f"ema_n_{i}"))
                    n.index_add_(0, idx, torch.ones_like(idx, dtype=n.dtype))
                    w = torch.zeros_like(getattr(self, f"ema_w_{i}"))
                    w.index_add_(0, idx, flat)
                    getattr(self, f"ema_n_{i}").mul_(self.decay).add_(
                        n, alpha=1 - self.decay)
                    getattr(self, f"ema_w_{i}").mul_(self.decay).add_(
                        w, alpha=1 - self.decay)
                    counts = getattr(self, f"ema_n_{i}").clamp_min(1e-3)
                    emb_i.weight.data.copy_(
                        getattr(self, f"ema_w_{i}") / counts.unsqueeze(1))
            hit = torch.zeros(self.vqvae_vocab_sizes[i], device=z.device)
            hit.index_add_(0, idx, torch.ones_like(idx, dtype=hit.dtype))
            per_util.append(float((hit > 0).sum()) / hit.numel())

            q_full = self._up_to_full(q, i)
            z_hat = z_hat + q_full
            z_rest = z_rest - q_full
            # commitment: pull encoder z toward quantized approx (stop-grad cb)
            vq_loss = vq_loss + self.commitment_cost * F.mse_loss(
                z_hat.detach(), z)

        vq_loss = vq_loss / K
        z_q = z + (z_hat - z).detach()        # straight-through for decoder
        total = sum(self.vqvae_vocab_sizes)
        gutil = sum(u * v for u, v in
                   zip(per_util, self.vqvae_vocab_sizes)) / total
        return z_q, vq_loss, per_util, gutil, 0.0

    # ---- encode / decode helpers -----------------------------------------

    @torch.no_grad()
    def z_to_idx(self, z):
        B, L, C = z.shape
        z_rest = z.clone()
        out = []
        K = len(self.levels)
        for i, L_i in enumerate(self.levels):
            emb_i = self.embeddings[i]
            inp = z_rest if i == K - 1 else self._interp(z_rest, L_i)
            flat = inp.reshape(-1, C)
            idx = self._nearest(flat, emb_i.weight).view(B, L_i)
            q = emb_i.weight[idx.view(-1)].view(B, L_i, C)
            z_rest = z_rest - self._up_to_full(q, i)
            out.append(idx)
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def idx_to_z_q(self, idx):
        B = idx.shape[0]
        z_hat = torch.zeros(B, self.L_full, self.embed_dim, device=idx.device)
        ofs = 0
        for i, L_i in enumerate(self.levels):
            q = self.embeddings[i].weight[idx[:, ofs:ofs + L_i].reshape(-1)
                                          ].view(B, L_i, self.embed_dim)
            z_hat = z_hat + self._up_to_full(q, i)
            ofs += L_i
        return z_hat

    @torch.no_grad()
    def idx_to_ncm_input(self, idx):
        """Input chunk for level k+1 = interp(accumulated z_hat of levels<=k)."""
        B = idx.shape[0]
        z_hat = torch.zeros(B, self.L_full, self.embed_dim, device=idx.device)
        chunks = []
        ofs = 0
        for i, L_i in enumerate(self.levels[:-1]):
            q = self.embeddings[i].weight[idx[:, ofs:ofs + L_i].reshape(-1)
                                          ].view(B, L_i, self.embed_dim)
            z_hat = z_hat + self._up_to_full(q, i)
            chunks.append(self._interp(z_hat, self.levels[i + 1]))
            ofs += L_i
        return torch.cat(chunks, dim=1)
