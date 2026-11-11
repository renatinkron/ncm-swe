#!/usr/bin/env python3
"""Train per-level pyramid heads on a FROZEN Qwen backbone (Medusa-1-style,
but heads predict RVQ pyramid codes instead of future tokens).

Pipeline: text → BPE tokens → backbone hidden states (frozen, cached) →
L small heads; head_k predicts pyramid code k conditioned on codes <k.
Only heads train (~tens of M params) — CPU-feasible.

--offsets K: train K independent LevelHeads sets (block drafting).
One forward over ids[:, :-1]; feature h[:, -1-j] feeds head set j, which
learns to predict the SAME final token ids[:, -1] — i.e. distance j+1
ahead (Medusa-style multi-offset). offsets=1 reproduces the original
next-token behaviour exactly.

Usage:
  python scripts/train_level_heads.py --model-dir /tmp/llm/qwen05 \
      --codebook artifacts/pyramid_cb_qwen05.pt --steps 2000 \
      --out artifacts/level_heads
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


# ---------- frozen Qwen backbone (minimal: embed + N blocks + norm) -------
class FrozenQwen(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        st = load_file(os.path.join(model_dir, "model.safetensors"))
        self.cfg = cfg
        nl, nh, nk = cfg["num_hidden_layers"], cfg["num_attention_heads"], \
            cfg["num_key_value_heads"]
        hd = cfg["hidden_size"] // nh
        self.hd, self.nh, self.nk = hd, nh, nk
        self.embed = nn.Embedding.from_pretrained(
            st["model.embed_tokens.weight"].float(), freeze=True)
        self.norm = nn.Parameter(st["model.norm.weight"].float(),
                                 requires_grad=False)
        blocks = []
        for i in range(nl):
            p = f"model.layers.{i}."
            blocks.append({
                "ln1": st[p + "input_layernorm.weight"].float(),
                "ln2": st[p + "post_attention_layernorm.weight"].float(),
                "q": st[p + "self_attn.q_proj.weight"].float(),
                "k": st[p + "self_attn.k_proj.weight"].float(),
                "v": st[p + "self_attn.v_proj.weight"].float(),
                "o": st[p + "self_attn.o_proj.weight"].float(),
                "gate": st[p + "mlp.gate_proj.weight"].float(),
                "up": st[p + "mlp.up_proj.weight"].float(),
                "down": st[p + "mlp.down_proj.weight"].float(),
            })
        self.blocks = [nn.ParameterDict({k: nn.Parameter(v, requires_grad=False)
                                         for k, v in b.items()})
                       for b in blocks]
        self.blocks = nn.ParameterList(self.blocks)

    def _rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

    def _rope(self, x, pos, base=1e6):
        B, H, L, D = x.shape
        inv = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device) / D))
        ang = pos[:, None] * inv[None]                     # L×D/2
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rot = torch.stack([-x2, x1], -1).flatten(-2)
        return x * torch.cat([cos, cos], -1) + rot * torch.cat([sin, sin], -1)

    @torch.no_grad()
    def forward(self, ids):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        x = self.embed(ids)
        for b in self.blocks:
            h = self._rms(x, b["ln1"])
            q = (h @ b["q"].t()).view(B, L, self.nh, self.hd).transpose(1, 2)
            k = (h @ b["k"].t()).view(B, L, self.nk, self.hd).transpose(1, 2)
            v = (h @ b["v"].t()).view(B, L, self.nk, self.hd).transpose(1, 2)
            q, k = self._rope(q, pos), self._rope(k, pos)
            k = k.repeat_interleave(self.nh // self.nk, 1)
            v = v.repeat_interleave(self.nh // self.nk, 1)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).reshape(B, L, -1) @ b["o"].t()
            x = x + att
            h = self._rms(x, b["ln2"])
            x = x + F.silu(h @ b["gate"].t()) * (h @ b["up"].t()) @ b["down"].t()
        return self._rms(x, self.norm)                    # B×L×H features


# ---------- level heads ----------------------------------------------------
class LevelHeads(nn.Module):
    """head_k: (feature, code_0..k-1 embeddings) -> logits over codebook_k."""
    def __init__(self, d_feat, codebooks):
        super().__init__()
        self.L = len(codebooks)
        self.codebooks = codebooks
        self.d_code = codebooks[0].shape[1]
        self.proj = nn.Linear(d_feat, 256)
        # cond[k] receives codes of levels <k ONLY (no leakage of the target
        # code k itself); head_0 sees just the projected feature
        self.cond = nn.ModuleList([
            nn.Sequential(nn.Linear(k * self.d_code, 256), nn.SiLU())
            if k else nn.Identity() for k in range(self.L)])
        self.out = nn.ModuleList([nn.Linear(256, cb.shape[0])
                                  for cb in codebooks])
        self.cw = nn.ParameterList([
            nn.Parameter(cb.clone(), requires_grad=False) for cb in codebooks])

    def forward(self, feat, prev_codes):
        """feat: B×d; prev_codes: true code indices (teacher forcing) —
        head_k consumes prev_codes[0..k-1] only."""
        h = self.proj(feat)
        outs = []
        for k in range(self.L):
            ctx = torch.cat([self.cw[j][prev_codes[j]] for j in range(k)],
                            -1) if k else h
            outs.append(self.out[k](self.cond[k](ctx) + (h if k else 0)))
        return outs


def codes_of_tokens(token_ids, codebook_idx):
    """token_ids B×L -> per-level code indices via codebook assignment."""
    return [[cb[token_ids] for cb in level] for level in codebook_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--corpus", default="artifacts/corpus")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--offsets", type=int, default=1,
                    help="K head sets: set j predicts token j+1 ahead "
                         "from the same backbone forward (default 1)")
    ap.add_argument("--out", default="artifacts/level_heads")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    cb = torch.load(a.codebook, weights_only=False)
    cbs = [c.float() for c in cb["codebooks"]]
    L = len(cbs)
    # token -> code path lookup table (V × L)
    print("[heads] building token→codes table…")
    from safetensors.torch import load_file as lf
    E = lf(os.path.join(a.model_dir, "model.safetensors"))[
        "model.embed_tokens.weight"].float()
    tok2codes = torch.stack(
        [torch.zeros(E.shape[0], dtype=torch.long) for _ in range(L)])
    res = E.clone()
    for li, C in enumerate(cbs):
        cn = C.pow(2).sum(1)
        for s in range(0, E.shape[0], 4096):
            xb = res[s:s + 4096]
            d = xb.pow(2).sum(1, keepdim=True) + cn - 2 * xb @ C.t()
            tok2codes[li][s:s + 4096] = d.argmin(1)
            res[s:s + 4096] -= C[d.argmin(1)]
    print("[heads] token→codes table built")

    model = FrozenQwen(a.model_dir).eval()
    heads = nn.ModuleList([LevelHeads(model.embed.weight.shape[1], cbs)
                           for _ in range(a.offsets)])
    nparams = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    print(f"[heads] trainable params: {nparams/1e6:.1f}M, levels={L}, "
          f"offsets={a.offsets}")

    # Qwen-token stream (pyramid indexes Qwen's vocab — must be Qwen ids)
    stream_path = "artifacts/qwen_stream.pt"
    if os.path.exists(stream_path):
        toks = torch.load(stream_path, weights_only=False)
    else:
        from transformers import AutoTokenizer
        import glob as _glob
        tk = AutoTokenizer.from_pretrained(a.model_dir)
        files = sorted(_glob.glob(os.path.join(a.corpus, "*.jsonl"))) \
            if os.path.isdir(a.corpus) else [a.corpus]
        ids = []
        for fp in files:
            with open(fp) as f:
                for i, line in enumerate(f):
                    if i % 500 == 0:
                        print(f"[heads] tokenizing {fp}:{i}…", flush=True)
                    ids.extend(tk(json.loads(line)["text"]).input_ids)
        toks = torch.tensor(ids, dtype=torch.long)
        torch.save(toks, stream_path)
        print(f"[heads] qwen stream saved: {toks.numel()} tokens")
    toks = toks[:toks.numel() // a.seq * a.seq].view(-1, a.seq)

    opt = torch.optim.AdamW([p for p in heads.parameters()
                             if p.requires_grad], lr=a.lr)
    start = 0
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, weights_only=False)
        heads.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        start = sd["step"]; print(f"[heads] resumed at step {start}")

    os.makedirs(a.out, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(start, a.steps):
        idx = torch.randint(0, toks.shape[0], (a.batch,), generator=g)
        ids = toks[idx]
        inp, tgt = ids[:, :-1], ids[:, 1:]
        h = model(inp)                                # B×(seq-1)×d all feats
        # per-position training would be stronger but costlier; last pos is
        # the practical decode context
        prev = [tok2codes[k][tgt[:, -1]] for k in range(L)]
        loss, per = 0.0, [0.0] * a.offsets
        for j in range(a.offsets):
            logits = heads[j](h[:, -1 - j], prev)
            l_j = sum(F.cross_entropy(logits[k], prev[k])
                      for k in range(L)) / L
            loss, per[j] = loss + l_j / a.offsets, float(l_j)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            print(f"[heads] step {step}/{a.steps} loss={float(loss):.4f} "
                  + " ".join(f"off{j}={v:.3f}" for j, v in enumerate(per)),
                  flush=True)
        if (step + 1) % 250 == 0:
            torch.save({"model": heads.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1}, ckpt_path)
    torch.save({"model": heads.state_dict(), "step": a.steps},
               os.path.join(a.out, "final.pt"))
    print(f"[heads] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
