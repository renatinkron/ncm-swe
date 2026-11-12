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
                 shared_head=True, intra=False, intra_passes=32,
                 intra_mask='boundary',
                 corrupt=0.0, corrupt_schedule='flat',
                 corrupt_hi=None, corrupt_lo=None):
        super().__init__()
        # NOTE: vqvae registered AFTER _init_weights below so apply() cannot
        # re-randomize the frozen pretrained tokenizer (found in RUN-017
        # blind review: assigning here scrambles every Linear/Embedding).
        self.intra = intra
        self.intra_passes = intra_passes
        assert intra_mask in ('boundary', 'random', 'mix'), \
            f"intra_mask inválido: {intra_mask}"
        self.intra_mask = intra_mask
        self.corrupt = corrupt
        self.corrupt_schedule = corrupt_schedule
        self.corrupt_hi = corrupt if corrupt_hi is None else corrupt_hi
        self.corrupt_lo = corrupt if corrupt_lo is None else corrupt_lo
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
        if intra:
            # per-level learned mask embedding in VQ z-space; zeros init so
            # pending positions start as pure coarse input (= parallel mode)
            self.mask_emb = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.embed_dim))
                 for _ in self.levels])
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
        self.vqvae = vqvae  # frozen tokenizer — registered post-init on purpose

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

    def _intra_add(self, idx, boundaries, decided=None):
        """(B,T,D) additive tensor for intra-level conditioning.

        For chunk i (predicting level i+1, target codes idx[:, ofs:ofs+L]):
        decided positions  -> + emb_{i+1}(code)
        pending positions  -> + mask_emb_{i+1}
        `decided`: optional list of (B,L) bool masks per chunk level —
        arbitrary patterns (Mask-Predict-style sparse remasking).
        `boundaries`: (B, n_chunks) prefix cut per level (left→right chunks).
        Both None -> all pending (no leak, matches pure-parallel)."""
        B = idx.shape[0]
        vq = self.vqvae.vq
        add = torch.zeros(B, self.T, self.embed_dim, device=idx.device)
        tgt_ofs = self.levels[0]
        seq_ofs = 0
        arange_cache = {}
        for i, L in enumerate(self.chunk_levels):
            codes_i = idx[:, tgt_ofs:tgt_ofs + L]
            ce = vq.embeddings[i + 1](codes_i)
            if decided is not None:
                dec = decided[i]
            elif boundaries is None:
                dec = torch.zeros(B, L, dtype=torch.bool, device=idx.device)
            else:
                if L not in arange_cache:
                    arange_cache[L] = torch.arange(L, device=idx.device)
                dec = arange_cache[L].unsqueeze(0) < boundaries[:, i:i + 1]
            add[:, seq_ofs:seq_ofs + L] = torch.where(
                dec.unsqueeze(-1), ce, self.mask_emb[i + 1].view(1, 1, -1))
            seq_ofs += L
            tgt_ofs += L
        return add

    def _corrupt_codes(self, idx):
        """Official-NCM-style conditioning corruption.

        schedule 'flat' (default/backward-compat): per-item rate
        r_i ~ U(0, self.corrupt); each code replaced w.p. r_i.
        schedule 'linear': per-LEVEL rate, linearly decreasing from
        corrupt_hi (first conditioned level) to corrupt_lo (last) —
        official recipe per WPS replication (strong arm 0.75→0.25).
        schedule 'self' (GLAT-lite): same linear rates, but each replaced
        code is sampled from the model's own current predictions for that
        level — corrompe com os erros reais do modelo, nao ruido uniforme
        (Qian et al. 2021, arXiv 2008.07905). Costs one extra forward.
        Targets stay clean; level 0 never corrupted."""
        out = idx.clone()
        B = idx.shape[0]
        sched = self.corrupt_schedule
        hi = self.corrupt_hi
        lo = self.corrupt_lo
        rates = torch.rand(B, 1, device=idx.device) * self.corrupt \
            if sched == 'flat' else None
        preds = None
        if sched == 'self':
            saved = self.corrupt
            was_training = self.training
            try:
                self.corrupt = 0.0
                self.eval()  # own predictions without dropout perturbation
                _, outs = self(idx)
            finally:
                self.corrupt = saved
                if was_training:
                    self.train()
            preds = [torch.multinomial(
                o.softmax(-1).reshape(-1, o.shape[-1]), 1
            ).reshape(B, -1) for o in outs]
            del outs
        ofs = self.levels[0]  # skip level 0 (official: skip_level0) — l0 seeds
                              # z_hat for every chunk; corrupting it poisons all
        nlv = len(self.levels) - 1
        for j, L in enumerate(self.levels[1:], start=1):
            if sched in ('linear', 'self'):
                # rate fixa por nível: hi → lo linearmente
                r = hi + (lo - hi) * (j - 1) / max(nlv - 1, 1)
                mask_lvl = torch.rand(B, L, device=idx.device) < r
            else:
                mask_lvl = (
                    torch.rand(idx.shape, device=idx.device) < rates
                )[:, ofs:ofs + L]
            rnd = preds[j - 1] if sched == 'self' else torch.randint(
                0, self.vocab_sizes[j], (B, L), device=idx.device)
            out[:, ofs:ofs + L] = torch.where(mask_lvl, rnd,
                                              out[:, ofs:ofs + L])
            ofs += L
        return out

    def forward(self, idx, boundaries=None, decided=None):
        """idx: (B, sum(levels)). Returns (logits_l0 (B,l0,V0), outs[i]
        (B,L_{i+1},V_{i+1})) for training."""
        B = idx.shape[0]
        with torch.no_grad():
            idx_in = self._corrupt_codes(idx) if self.corrupt > 0 else idx
            x = self.vqvae.vq.idx_to_ncm_input(idx_in)  # (B,T,D_vq)
        if self.intra:
            x = x + self._intra_add(idx_in, boundaries, decided)
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

    def _sample_intra_masks(self, idx):
        """Returns (boundaries, decided) for intra training per intra_mask:
        'boundary' — per-item prefix cut ~ U{0..L} per level (original).
        'random'   — per-item decided SUBSET of size k ~ U{0..L} per level
                     (arbitrary patterns; enables Mask-Predict refinement).
        'mix'      — 50/50 boundary and random per item per level."""
        B, dev = idx.shape[0], idx.device
        if self.intra_mask == 'boundary':
            return torch.stack(
                [torch.randint(0, L + 1, (B,), device=dev)
                 for L in self.chunk_levels], dim=1), None
        decided = []
        for L in self.chunk_levels:
            k = torch.randint(0, L + 1, (B, 1), device=dev)
            perm = torch.argsort(torch.rand(B, L, device=dev), dim=1)
            rank = torch.empty_like(perm)
            rank.scatter_(1, perm, torch.arange(L, device=dev)
                          .unsqueeze(0).expand(B, L))
            dec = rank < k
            if self.intra_mask == 'mix':
                b = torch.randint(0, L + 1, (B, 1), device=dev)
                pref = (torch.arange(L, device=dev).unsqueeze(0) < b)
                dec = torch.where(torch.rand(B, 1, device=dev) < 0.5,
                                  pref, dec)
            decided.append(dec)
        return None, decided

    def loss(self, idx):
        boundaries = decided = None
        if self.intra:
            boundaries, decided = self._sample_intra_masks(idx)
        logits_l0, outs = self.forward(idx, boundaries, decided)
        l0 = self.levels[0]
        t0 = idx[:, :l0]
        loss_l0 = F.cross_entropy(
            logits_l0.reshape(-1, logits_l0.shape[-1]), t0.reshape(-1))
        total = loss_l0
        ofs = l0
        per_level = {"l0": round(float(loss_l0.detach()), 4)}
        for i, out in enumerate(outs):
            L = self.chunk_levels[i]
            t = idx[:, ofs:ofs + L]
            if self.intra:
                if decided is not None:
                    pending = ~decided[i]
                else:
                    pending = (torch.arange(L, device=idx.device)
                               .unsqueeze(0) >= boundaries[:, i:i + 1])
                ce_map = F.cross_entropy(
                    out.reshape(-1, out.shape[-1]), t.reshape(-1),
                    reduction="none").reshape(t.shape)
                ce = ((ce_map * pending).sum()
                      / pending.sum().clamp_min(1))
            else:
                ce = F.cross_entropy(
                    out.reshape(-1, out.shape[-1]), t.reshape(-1))
            per_level[f"l{i+1}"] = round(float(ce.detach()), 4)
            total = total + ce
            ofs += L
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
                if hasattr(vq, "_interp"):
                    chunks.append(vq._interp(z_hat, self.levels[j + 1]))
                else:
                    chunks.append(F.interpolate(
                        z_hat.transpose(-1, -2), size=(self.levels[j + 1],),
                        mode="area").transpose(-1, -2))
        return chunks  # chunks[k] predicts level k+1

    def _forward_partial(self, codes_by_level, cur=None, decided=None):
        """Run the stack over [q0] + chunks predicting levels 1..len(codes).

        Intra mode: `cur`/`decided` (B,L_k) inject emb(code) at decided
        positions of the level being predicted, mask_emb elsewhere."""
        B = codes_by_level[0].shape[0]
        k_next = len(codes_by_level)  # predict level index k_next
        chunks = self._chunks_from_codes(codes_by_level, k_next)
        if self.intra and cur is not None:
            L = self.levels[k_next]
            ce = self.vqvae.vq.embeddings[k_next](cur)
            chunks[-1] = chunks[-1] + torch.where(
                decided.unsqueeze(-1), ce, self.mask_emb[k_next].view(1, 1, -1))
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
                 return_per_level=False, l0=None,
                 refine_frac=0.0, refine_iters=0, refine_from=1):
        """Coarse→fine generation in K passes. Returns idx (B,sum(levels)).

        l0: optional (B, levels[0]) long tensor — external level-0 codes
        (official recipe samples l0 ~ corpus unigram instead of predicting).
        l0=None preserves the default: l0 is sampled from the model head.

        Mask-Predict refinement (intra only): after each level k >=
        refine_from, run refine_iters rounds — remask the refine_frac
        positions whose sampled code had the LOWEST probability when it was
        predicted while pending, then re-predict pending positions in
        parallel. Confidence is captured at prediction time (positions are
        genuinely pending then) — scoring under all-decided context would
        be self-referential since position t's input already carries
        emb(code_t). Each round costs 1 transformer pass.
        refine_iters=0 disables (default)."""
        self.eval()
        codes = []
        if l0 is not None:
            l0 = l0.to(self.q0.device)
            assert l0.shape == (batch_size, self.levels[0]), \
                f"l0 shape {tuple(l0.shape)} != ({batch_size},{self.levels[0]})"
            v0 = int(self.vocab_sizes[0])
            assert int(l0.min()) >= 0 and int(l0.max()) < v0, \
                f"l0 fora do range [0,{v0})"
            codes.append(l0)
        else:
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
        passes = 0
        for k in range(1, len(self.levels)):
            Lk = self.levels[k]
            if not self.intra:
                h = self._forward_partial(codes)
                logits = self._head_logits(h[:, -Lk:], k)
                codes.append(self._sample(
                    logits.reshape(batch_size * Lk, -1), temperature,
                    top_k, top_p).reshape(batch_size, Lk))
                passes += 1
                continue
            # intra: causal chunks within the level (left→right)
            cur = torch.zeros(batch_size, Lk, dtype=torch.long,
                              device=self.q0.device)
            decided = torch.zeros(batch_size, Lk, dtype=torch.bool,
                                  device=self.q0.device)
            conf = torch.zeros(batch_size, Lk, device=self.q0.device)
            for s in range(0, Lk, self.intra_passes):
                h = self._forward_partial(codes, cur, decided)
                lg = self._head_logits(h[:, -Lk:], k)[:, s:s + self.intra_passes]
                codes_s = self._sample(
                    lg.reshape(-1, lg.shape[-1]), temperature, top_k,
                    top_p).reshape(batch_size, -1)
                w = codes_s.shape[1]
                conf[:, s:s + w] = F.softmax(lg, -1).gather(
                    -1, codes_s.unsqueeze(-1)).squeeze(-1)
                cur[:, s:s + w] = codes_s
                decided[:, s:s + w] = True
                passes += 1
            if refine_iters > 0 and k >= refine_from:
                n_mask = max(1, int(round(Lk * refine_frac)))
                for _ in range(refine_iters):
                    low = conf.argsort(dim=1)[:, :n_mask]
                    decided2 = torch.ones(batch_size, Lk, dtype=torch.bool,
                                          device=self.q0.device)
                    decided2.scatter_(1, low, False)
                    h = self._forward_partial(codes, cur, decided2)
                    lg = self._head_logits(h[:, -Lk:], k)
                    probs = F.softmax(lg, -1)
                    new = self._sample(
                        lg.reshape(-1, lg.shape[-1]), temperature,
                        top_k, top_p).reshape(batch_size, Lk)
                    new_conf = probs.gather(
                        -1, new.unsqueeze(-1)).squeeze(-1)
                    cur = torch.where(decided2, cur, new)
                    conf = torch.where(decided2, conf, new_conf)
                    passes += 1
            codes.append(cur)
        idx = torch.cat(codes, dim=1)
        self.last_passes = passes + (1 if l0 is None else 0)
        if return_per_level:
            return idx, codes
        return idx

    @torch.no_grad()
    def decode_text(self, idx, bpe):
        logits = self.vqvae.decode(idx.to(next(self.parameters()).device))
        ids = logits.argmax(-1)
        return [bpe.decode(row.tolist(), skip_special_tokens=True)
                for row in ids]
