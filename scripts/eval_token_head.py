#!/usr/bin/env python3
"""Held-out eval for token heads (RUN-010 protocol, persisted).

Evaluates both variants (feat / codes) of a trained TokenHead
checkpoint on the deterministic held-out slice of qwen_stream.pt
(last 5% of rows, seq=64) — the same split used in training.

Metrics per variant over --positions rows:
  t1   — top-1 vs the true corpus token (row[-1])
  a1   — top-1 vs the frozen backbone's AR argmax (feat @ E.T)
  rank — mean rank of the AR argmax token under the head's
         distribution (1 = perfect; diagnostic from RUN-009)

Usage:
  python scripts/eval_token_head.py --model-dir ~/models/qwen05-sft \
      --head artifacts/token_head_ar_qwen05/final.pt \
      --positions 128 --out artifacts/token_head_ar_eval.json
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_token_head import (FrozenQwen, LevelHeads, TokenHead,  # noqa: E402
                              predicted_code_vecs, accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--codebook", default="artifacts/pyramid_cb_qwen05.pt")
    ap.add_argument("--heads", default="artifacts/level_heads_qwen05/final.pt")
    ap.add_argument("--head", required=True,
                    help="token_head checkpoint dir/final.pt to eval")
    ap.add_argument("--stream", default="artifacts/qwen_stream.pt")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--positions", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from safetensors.torch import load_file
    E = load_file(os.path.join(a.model_dir, "model.safetensors"))[
        "model.embed_tokens.weight"].float()
    V, d = E.shape
    cb = torch.load(a.codebook, weights_only=False)
    cbs = [c.float() for c in cb["codebooks"]]
    L = len(cbs)
    model = FrozenQwen(a.model_dir).eval()
    heads = LevelHeads(d, cbs)
    heads.load_state_dict(
        torch.load(a.heads, weights_only=False)["model"])
    heads.eval()
    variants = torch.nn.ModuleDict({
        "feat": TokenHead(d, V), "codes": TokenHead(d + L * d, V)})
    sd = torch.load(a.head, weights_only=False)
    variants.load_state_dict(sd["model"])
    variants.eval()
    print(f"[eval] loaded {a.head} (step={sd.get('step')})")

    toks = torch.load(a.stream, weights_only=False)
    toks = toks[:toks.numel() // a.seq * a.seq].view(-1, a.seq)
    hold = toks[-int(toks.shape[0] * 0.05):]
    N = min(a.positions, hold.shape[0])
    res = {k: {"t1": 0, "a1": 0, "rank_sum": 0.0}
           for k in variants}
    with torch.no_grad():
        for i in range(0, N, 32):
            ids = hold[i:i + 32]
            inp, tgt = ids[:, :-1], ids[:, -1]
            feat = model(inp)[:, -1]
            ar_tok = (feat @ E.t()).argmax(-1)
            cvecs = predicted_code_vecs(heads, feat, cbs)
            for name, th in variants.items():
                x = feat if name == "feat" else torch.cat(
                    [feat, cvecs], -1)
                logits = th.net(x)
                t1, a1 = accs(logits, tgt, ar_tok)
                res[name]["t1"] += t1
                res[name]["a1"] += a1
                order = logits.argsort(-1, descending=True)
                ranks = (order == ar_tok.unsqueeze(-1)).nonzero(
                    as_tuple=True)[1] + 1
                res[name]["rank_sum"] += float(ranks.float().sum())
    out = {k: {"t1": v["t1"] / N, "a1": v["a1"] / N,
               "mean_ar_rank": v["rank_sum"] / N}
           for k, v in res.items()}
    out["n"] = N
    out["head"] = a.head
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[eval] -> {a.out}")


if __name__ == "__main__":
    main()
