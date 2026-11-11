#!/usr/bin/env python3
"""Adaptive pyramid decode: Qwen backbone (frozen) + trained level heads +
RVQ codebook → token, with early-exit on level confidence.

The "adaptive supersampling" experiment (RUN-005 T003 / RUN-006 T004):
render coarse first (l0), refine only while uncertain. Measures the
speed/quality trade of depth vs a same-backbone AR baseline (native argmax
over tied embeddings — the honest control).

Modes:
  fixed    — always evaluate all L levels per position
  adaptive — exit after level k when min(level top-1 probs) > tau
             (cheap proxy: code-probability product, no embedding lookup
             until commit)
  ar       — backbone argmax baseline (no heads/codebook)

Metrics per mode: tokens/s, mean levels used, token agreement vs AR argmax,
per-level usage histogram. Output: artifacts/pyramid_bench.json

Usage:
  python scripts/decode_pyramid.py --model-dir ~/models/qwen05-sft \
      --codebook artifacts/pyramid_cb_qwen05.pt \
      --heads artifacts/level_heads_qwen05/final.pt \
      --prompt "def quicksort" --n 32 --mode adaptive --tau 0.6
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_level_heads import FrozenQwen, LevelHeads  # noqa: E402


def nearest_token(recon, E, chunk=8192):
    """recon (d,) -> argmin ||recon - E[v]|| over vocab. Chunked to bound RAM."""
    best_d, best_i = None, -1
    cn = E.pow(2).sum(1)                       # V
    for s in range(0, E.shape[0], chunk):
        d = cn[s:s + chunk] - 2 * (E[s:s + chunk] @ recon)
        m, i = d.min(0)
        if best_d is None or m < best_d:
            best_d, best_i = m, s + int(i)
    return best_i, float(best_d)


def pyramid_token(heads, feat, cbs, E, mode, tau):
    """One position: level-by-level codes -> reconstructed embedding ->
    nearest token. Returns (token_id, levels_used, conf_list)."""
    prev_vecs, confs, used = [], [], 0
    h = heads.proj(feat)
    for k in range(heads.L):
        if k == 0:
            logits = heads.out[0](h)
        else:
            ctx = torch.cat(prev_vecs, -1)
            logits = heads.out[k](heads.cond[k](ctx) + h)
        p = F.softmax(logits, -1)
        conf, c = p.max(-1)
        confs.append(float(conf))
        prev_vecs.append(cbs[k][int(c)])
        used = k + 1
        if mode == "adaptive" and min(confs) > tau:
            break
    recon = sum(v for v in prev_vecs)
    tok, dist = nearest_token(recon, E)
    return tok, used, confs, int(c)


def gen_pyramid(model, heads, cbs, E, ids, n, mode, tau, ctx):
    """Autoregressive over positions; pyramid decode inside each position."""
    ids = ids.clone()
    stats = []
    t0 = time.time()
    for _ in range(n):
        feat = model(ids[:, -ctx:])[0, -1]
        tok, used, confs, _ = pyramid_token(heads, feat, cbs, E, mode, tau)
        ids = torch.cat([ids, torch.tensor([[tok]])], 1)
        stats.append(used)
    dt = time.time() - t0
    return ids, stats, dt


def gen_ar(model, E, ids, n, ctx):
    """Same-backbone AR baseline: argmax over tied embeddings (native logits)."""
    ids = ids.clone()
    t0 = time.time()
    for _ in range(n):
        feat = model(ids[:, -ctx:])[0, -1]
        tok = int((feat @ E.t()).argmax())
        ids = torch.cat([ids, torch.tensor([[tok]])], 1)
    return ids, time.time() - t0


def agreement(model, E, gen_ids, plen, ctx):
    """Fraction of generated tokens (positions >= plen) equal to the AR
    argmax computed on the same trajectory (prefix = all tokens < pos)."""
    same = 0
    n = gen_ids.shape[1] - plen
    for pos in range(plen, gen_ids.shape[1]):
        feat = model(gen_ids[:, :pos][:, -ctx:])[0, -1]
        if int((feat @ E.t()).argmax()) == int(gen_ids[0, pos]):
            same += 1
    return same / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--codebook", default="artifacts/pyramid_cb_qwen05.pt")
    ap.add_argument("--heads", default="artifacts/level_heads_qwen05/final.pt")
    ap.add_argument("--prompt", default="def fibonacci")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--mode", default="adaptive",
                    choices=["fixed", "adaptive", "ar", "sweep"])
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--out", default="artifacts/pyramid_bench.json")
    a = ap.parse_args()

    from safetensors.torch import load_file
    E = load_file(os.path.join(a.model_dir, "model.safetensors"))[
        "model.embed_tokens.weight"].float()
    cb = torch.load(a.codebook, weights_only=False)
    cbs = [c.float() for c in cb["codebooks"]]
    model = FrozenQwen(a.model_dir).eval()
    heads = LevelHeads(model.embed.weight.shape[1], cbs)
    sd = torch.load(a.heads, weights_only=False)
    heads.load_state_dict(sd["model"])
    heads.eval()
    print(f"[decode] loaded: V={E.shape[0]} d={E.shape[1]} levels="
          f"{[c.shape[0] for c in cbs]}")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model_dir)
    ids = torch.tensor([tk(a.prompt).input_ids])
    print(f"[decode] prompt: {a.prompt!r} -> {ids.shape[1]} tokens")

    out = {"config": vars(a), "runs": {}}
    if a.mode == "sweep":
        modes = ["ar", "fixed"] + [f"adaptive_t{t}" for t in
                                   (0.3, 0.6, 0.9)]
    else:
        modes = [a.mode]
    for m in modes:
        tau = float(m.split("_t")[1]) if "_t" in m else a.tau
        m_mode = "adaptive" if "_t" in m else m
        if m == "ar":
            gen, dt = gen_ar(model, E, ids, a.n, a.ctx)
            new = gen[0, ids.shape[1]:].tolist()
            ag = agreement(model, E, gen, ids.shape[1], a.ctx)
            out["runs"]["ar"] = {
                "tok_per_s": a.n / dt, "wall_s": dt,
                "agree_with_argmax": ag,
                "text": tk.decode(new)}
            print(f"[ar] {a.n} tok in {dt:.1f}s = {a.n/dt:.2f} tok/s "
                  f"(self-agree {ag:.2f})")
        else:
            gen, stats, dt = gen_pyramid(model, heads, cbs, E, ids, a.n,
                                         m_mode, tau, a.ctx)
            new = gen[0, ids.shape[1]:].tolist()
            ag = agreement(model, E, gen, ids.shape[1], a.ctx)
            hist = {i + 1: stats.count(i + 1) for i in range(len(cbs))}
            out["runs"][m] = {
                "tok_per_s": a.n / dt, "wall_s": dt,
                "mean_levels": sum(stats) / len(stats),
                "level_hist": hist,
                "agree_with_argmax": ag,
                "text": tk.decode(new)}
            print(f"[{m}] {a.n} tok in {dt:.1f}s = {a.n/dt:.2f} tok/s "
                  f"mean_lvl={sum(stats)/len(stats):.1f} agree={ag:.2f} "
                  f"hist={hist}")

    torch.save({"stats": out}, "/tmp/_last_dec.pt")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[decode] -> {a.out}")


if __name__ == "__main__":
    main()
