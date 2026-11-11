#!/usr/bin/env python3
"""Code-path likelihood decoding: token chosen by the JOINT probability
of its quantized code path, not by embedding distance.

For each vocab token v: score(v) = Σ_k log P(code_k(v) | feat,
code_{<k}(v)) — heads were trained with teacher forcing on true codes,
so every level's conditional for token v uses v's OWN earlier codes:
the whole distribution is parallel over the vocabulary.

Modes:
  full — exact Σ over all levels for all V tokens (chunked; costly at
         deep levels: out_5 is V×256×16384 per position)
  beam — after `--beam-after` levels keep only top-B tokens; much
         cheaper, approximate

Metrics per position: top-1 agreement of path-argmax vs backbone AR
argmax and vs nearest-embedding of the greedy code path (the RUN-006
baseline), plus top-5 hit for AR argmax. Positions = last-position
features of held-out stream rows (matches training context).

Usage:
  python scripts/codepath_decode.py --model-dir ~/models/qwen05-sft \
      --heads artifacts/level_heads_qwen05/final.pt \
      --mode full --positions 50
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_level_heads import FrozenQwen, LevelHeads  # noqa: E402
from decode_pyramid import nearest_token              # noqa: E402


def build_tok2codes(E, cbs, cache="artifacts/tok2codes.pt"):
    """V×L table of each token's code per level (residual assignment).
    Cached — rebuilds cost ~1min over 152k×896."""
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    V, L = E.shape[0], len(cbs)
    t2c = torch.stack([torch.zeros(V, dtype=torch.long) for _ in range(L)], 1)
    res = E.clone()
    for li, C in enumerate(cbs):
        cn = C.pow(2).sum(1)
        for s in range(0, V, 4096):
            d = cn[None, :] - 2 * res[s:s + 4096] @ C.t()
            c = d.argmin(1)
            t2c[s:s + 4096, li] = c
            res[s:s + 4096] -= C[c]
        print(f"[t2c] level {li} assigned", flush=True)
    torch.save(t2c, cache)
    return t2c


def path_scores(heads, feat, t2c, mode="full", beam_after=2, beam=256,
                vchunk=8192):
    """feat (d,) → (V_scores, kept_idx or None)."""
    h = heads.proj(feat)
    V, L = t2c.shape
    scores = torch.zeros(V)
    keep = None                                    # None = all tokens
    for k in range(L):
        if mode == "beam" and k >= beam_after:
            if keep is None:
                keep = scores.topk(beam).indices.sort().values
            idx = keep
        else:
            idx = torch.arange(V)
        codes = t2c[idx]                           # S×L subset
        for s in range(0, codes.shape[0], vchunk):
            cb_ = codes[s:s + vchunk]
            if k == 0:
                ctx = h.expand(cb_.shape[0], -1)
            else:
                prev = torch.cat([heads.cw[j][cb_[:, j]]
                                  for j in range(k)], -1)
                ctx = heads.cond[k](prev) + h
            logits = heads.out[k](ctx)
            logp = F.log_softmax(logits, -1)
            got = logp.gather(1, cb_[:, k:k + 1]).squeeze(1)
            scores[idx[s:s + vchunk]] += got
    return scores, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--codebook", default="artifacts/pyramid_cb_qwen05.pt")
    ap.add_argument("--heads", default="artifacts/level_heads_qwen05/final.pt")
    ap.add_argument("--stream", default="artifacts/qwen_stream.pt")
    ap.add_argument("--positions", type=int, default=50)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--mode", choices=["full", "beam"], default="full")
    ap.add_argument("--beam-after", type=int, default=2)
    ap.add_argument("--beam", type=int, default=256)
    ap.add_argument("--held-out-frac", type=float, default=0.05)
    ap.add_argument("--out", default="artifacts/codepath_eval.json")
    a = ap.parse_args()

    from safetensors.torch import load_file
    E = load_file(os.path.join(a.model_dir, "model.safetensors"))[
        "model.embed_tokens.weight"].float()
    cb = torch.load(a.codebook, weights_only=False)
    cbs = [c.float() for c in cb["codebooks"]]
    t2c = build_tok2codes(E, cbs)
    model = FrozenQwen(a.model_dir).eval()
    heads = LevelHeads(model.embed.weight.shape[1], cbs)
    heads.load_state_dict(torch.load(a.heads, weights_only=False)["model"])
    heads.eval()
    print(f"[cpd] V={E.shape[0]} L={len(cbs)} t2c={tuple(t2c.shape)}")

    toks = torch.load(a.stream, weights_only=False)
    toks = toks[:toks.numel() // a.seq * a.seq].view(-1, a.seq)
    n_hold = max(int(toks.shape[0] * a.held_out_frac), a.positions)
    hold = toks[-n_hold:]
    g = torch.Generator().manual_seed(1)
    rows = hold[torch.randperm(hold.shape[0], generator=g)[:a.positions]]
    print(f"[cpd] {a.positions} held-out positions (last {n_hold} rows)")

    res = {"config": vars(a), "per_pos": [], "summary": {}}
    hit_p = hit_n = top5 = 0
    t_start = time.time()
    for i, row in enumerate(rows):
        ids = row[:-1].unsqueeze(0)
        truth = int(row[-1])
        feat = model(ids)[0, -1]
        ar_tok = int((feat @ E.t()).argmax())
        ar_top5 = set((feat @ E.t()).topk(5).indices.tolist())
        # nearest-embedding baseline (greedy path through heads)
        h = heads.proj(feat); prev_vecs = []
        for k in range(heads.L):
            if k == 0:
                logits = heads.out[0](h)
            else:
                ctx = torch.cat(prev_vecs, -1)
                logits = heads.out[k](heads.cond[k](ctx) + h)
            c = int(logits.argmax(-1))
            prev_vecs.append(cbs[k][c])
        near_tok, _ = nearest_token(sum(prev_vecs), E)
        # code-path scoring
        t0 = time.time()
        scores, keep = path_scores(heads, feat, t2c, a.mode,
                                   a.beam_after, a.beam)
        path_tok = int(scores.argmax())
        dt = time.time() - t0
        hit_p += int(path_tok == ar_tok)
        hit_n += int(near_tok == ar_tok)
        top5 += int(path_tok in ar_top5)
        res["per_pos"].append({"truth": truth, "ar": ar_tok,
                               "path": path_tok, "near": near_tok,
                               "score_s": dt})
        if i % 10 == 0:
            print(f"[cpd] pos {i}: path={path_tok} ar={ar_tok} "
                  f"near={near_tok} ({dt:.1f}s)", flush=True)
    n = len(rows)
    res["summary"] = {
        "n_positions": n,
        "path_top1_vs_ar": hit_p / n, "near_top1_vs_ar": hit_n / n,
        "path_in_ar_top5": top5 / n,
        "mean_score_s": sum(p["score_s"] for p in res["per_pos"]) / n,
        "wall_s": time.time() - t_start}
    print(json.dumps(res["summary"], indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("[cpd] ->", a.out)


if __name__ == "__main__":
    main()
