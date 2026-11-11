#!/usr/bin/env python3
"""Hierarchical speculative decoding: block-pyramid heads draft K tokens
from ONE backbone feature; a single verification forward accepts the
longest AR-argmax-consistent prefix (Medusa structure, RVQ drafts).

Semantics (matches train_level_heads.py --offsets K): head set j consumes
the last-position feature and predicts pyramid codes of the token j+1
positions ahead. Draft = greedy codes over the first `--depth` levels →
partial reconstruction → nearest vocab embedding. Verify = one forward
over context+drafts; drafts matching the backbone's own argmax are
accepted, the first mismatch is corrected by the argmax for free, and a
full-match round also emits the bonus token.

Because emitted tokens always equal the backbone's greedy argmax, spec
output is IDENTICAL to pure-AR output — the contract property checked in
the bench.

Cost model: 1 forward per round (verify), except full-accept rounds need
one extra forward (hidden after the bonus token isn't computed). Speedup
≈ mean accepted tokens / forwards when backbone cost dominates.

Metrics persisted to artifacts/spec_pyramid_bench.json: tok/s both modes,
forwards, mean tokens/forward, per-offset acceptance, text equality.

Usage:
  python scripts/spec_pyramid.py --model-dir ~/models/qwen05-sft \
      --heads artifacts/block_heads_qwen05/final.pt \
      --prompt "def fibonacci" --n 32 --depth 1
"""
import argparse, json, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_level_heads import FrozenQwen, LevelHeads  # noqa: E402
from decode_pyramid import nearest_token              # noqa: E402
from codepath_decode import path_scores               # noqa: E402


tokhead = None                                # set in main for tokhead


def draft_token(head, feat, cbs, E, depth, t2c=None, score="embed",
                topk=1):
    """Greedy codes over `depth` levels → (top1, cand_set).
    cand_set is a frozenset of up to `topk` token ids (greedy
    k-candidate set, SpecInfer-style); only `tokhead` score supports
    topk>1 today — other scores return the singleton {top1}."""
    if score == "tokhead":
        from train_token_head import predicted_code_vecs
        cv = predicted_code_vecs(head, feat, cbs).unsqueeze(0)
        x = torch.cat([feat.unsqueeze(0), cv], -1)
        vals, idx = tokhead.net(x).topk(max(topk, 1), dim=-1)
        cand = frozenset(idx[0].tolist())
        return int(idx[0, 0]), cand if topk > 1 else None
    if score == "path":
        s, _ = path_scores(head, feat, t2c, "beam", depth, 512)
        return int(s.argmax()), None
    h = head.proj(feat)
    prev_vecs = []
    for k in range(depth):
        if k == 0:
            logits = head.out[0](h)
        else:
            ctx = torch.cat(prev_vecs, -1)
            logits = head.out[k](head.cond[k](ctx) + h)
        c = int(logits.argmax(-1))
        prev_vecs.append(cbs[k][c])
    tok, _ = nearest_token(sum(prev_vecs), E)
    return tok, None


def gen_spec(model, heads, cbs, E, ids, n, ctx, depth, t2c=None,
             score="embed", topk=1):
    """ids B=1 × L → +n tokens via block draft + AR verify."""
    K = len(heads)
    ids = ids.clone()
    t0, forwards = time.time(), 0
    accepted_hist = [0] * (K + 1)         # how many rounds accepted m tokens
    offset_ok = [0] * K                   # drafts matching when reached
    offset_seen = [0] * K
    frontier_hits = 0                     # accepted via topk set (not top1)
    feat = None                           # carried from verify when usable
    made = 0
    while made < n:
        if feat is None:                  # draft forward (first / full-accept)
            h = model(ids[:, -ctx:])
            forwards += 1
            feat = h[0, -1]
        drafts, cand_sets = [], []
        for j in range(K):
            t, cs = draft_token(heads[j], feat, cbs, E, depth, t2c,
                                score, topk)
            drafts.append(t)
            cand_sets.append(cs)
        ver = torch.cat([ids, torch.tensor([drafts])], 1)
        c = ids.shape[1]
        hv = model(ver[:, -ctx:])
        forwards += 1
        w = hv.shape[1]
        base = w - 1 - K                  # window pos of last real ctx token
        # hidden at base+j predicts ver index c+j → ar_j = argmax there;
        # ar[K] (pos w-1) is the bonus prediction for full-accept rounds
        ar = [int((hv[0, base + j] @ E.t()).argmax()) for j in range(K + 1)]
        m = 0
        frontier = False
        for j in range(K):
            offset_seen[j] += 1
            if drafts[j] == ar[j]:
                offset_ok[j] += 1
                m += 1
            elif (cand_sets[j] is not None and ar[j] in cand_sets[j]):
                # greedy k-candidate: argmax AR ∈ draft set at frontier —
                # emit ar[j] (≠draft top1) so deeper hiddens are stale:
                offset_ok[j] += 1
                m += 1
                frontier = True
                frontier_hits += 1
                break
            else:
                break
        if frontier:
            accepted = ar[:m]             # no bonus — context diverged
            feat = None                   # hidden after emitted ≠ top1
        else:
            accepted = drafts[:m] + [ar[m]]  # prefix + free correction
            if m == K:                    # all matched: next feat needs
                feat = None               # hidden after index c+K missing
            else:                         # hidden after corrected token
                feat = hv[0, base + m + 1]  # window pos after index c+m
        ids = torch.cat([ids, torch.tensor([accepted])], 1)
        accepted_hist[m] += 1
        made += len(accepted) if frontier else m + 1
    dt = time.time() - t0
    stats = {
        "forwards": forwards,
        "accepted_hist": accepted_hist,   # index m = drafts accepted (0..K)
        "mean_tok_per_fwd": made / max(forwards, 1),
        "accept_per_offset": [
            round(offset_ok[j] / max(offset_seen[j], 1), 3)
            for j in range(K)],
        "frontier_hits": frontier_hits,
    }
    return ids[:, :ids.shape[1] - (made - n)], stats, dt


def gen_ar(model, E, ids, n, ctx):
    ids = ids.clone()
    t0, forwards = time.time(), 0
    for _ in range(n):
        feat = model(ids[:, -ctx:])[0, -1]
        forwards += 1
        tok = int((feat @ E.t()).argmax())
        ids = torch.cat([ids, torch.tensor([[tok]])], 1)
    return ids, {"forwards": forwards}, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--codebook", default="artifacts/pyramid_cb_qwen05.pt")
    ap.add_argument("--heads", default="artifacts/block_heads_qwen05/final.pt")
    ap.add_argument("--prompt", default="def fibonacci(n):")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--depth", type=int, default=1,
                    help="pyramid levels used per draft token (1 = l0 only)")
    ap.add_argument("--score", choices=["embed", "path", "tokhead"],
                    default="embed",
                    help="embed: nearest embedding; path: code-path "
                         "likelihood argmax (needs tok2codes)")
    ap.add_argument("--token-head",
                    default="artifacts/token_head_qwen05/final.pt")
    ap.add_argument("--topk", type=int, default=1,
                    help="greedy k-candidate set size for tokhead draft "
                         "(SpecInfer-style); verify accepts frontier pos "
                         "if AR argmax in set")
    ap.add_argument("--out", default="artifacts/spec_pyramid_bench.json")
    a = ap.parse_args()

    from safetensors.torch import load_file
    E = load_file(os.path.join(a.model_dir, "model.safetensors"))[
        "model.embed_tokens.weight"].float()
    cb = torch.load(a.codebook, weights_only=False)
    cbs = [c.float() for c in cb["codebooks"]]
    model = FrozenQwen(a.model_dir).eval()
    sd = torch.load(a.heads, weights_only=False)
    K = len({k.split(".")[0] for k in sd["model"] if k.split(".")[0].isdigit()})
    heads = torch.nn.ModuleList(
        [LevelHeads(model.embed.weight.shape[1], cbs) for _ in range(K)])
    heads.load_state_dict(sd["model"])
    heads.eval()
    print(f"[spec] loaded: V={E.shape[0]} K={K} levels="
          f"{[c.shape[0] for c in cbs]} depth={a.depth}")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model_dir)
    ids = torch.tensor([tk(a.prompt).input_ids])

    out = {"config": vars(a), "runs": {}}
    gen, s_ar, dt = gen_ar(model, E, ids, a.n, a.ctx)
    ar_text = tk.decode(gen[0, ids.shape[1]:].tolist())
    out["runs"]["ar"] = {"tok_per_s": a.n / dt, "wall_s": dt,
                         "forwards": s_ar["forwards"], "text": ar_text}
    print(f"[ar] {a.n} tok in {dt:.1f}s = {a.n/dt:.2f} tok/s "
          f"({s_ar['forwards']} fwds)")
    t2c = None
    if a.score == "path":
        from codepath_decode import build_tok2codes
        t2c = build_tok2codes(E, cbs)
    global tokhead
    if a.score == "tokhead":
        from train_token_head import TokenHead
        tokhead = TokenHead(E.shape[1] + len(cbs) * E.shape[1],
                            E.shape[0])
        sd_t = torch.load(a.token_head, weights_only=False)["model"]
        tokhead.load_state_dict(
            {k[len("codes."):]: v for k, v in sd_t.items()
             if k.startswith("codes.")})
        tokhead.eval()
        print(f"[spec] tokhead(codes) loaded from {a.token_head}")
    gen, s_sp, dt = gen_spec(model, heads, cbs, E, ids, a.n, a.ctx,
                             a.depth, t2c, a.score, a.topk)
    sp_text = tk.decode(gen[0, ids.shape[1]:].tolist())
    out["runs"]["spec"] = {
        "tok_per_s": a.n / dt, "wall_s": dt, "text": sp_text,
        "text_equal_ar": sp_text == ar_text, **s_sp}
    print(f"[spec] {a.n} tok in {dt:.1f}s = {a.n/dt:.2f} tok/s "
          f"{s_sp['forwards']} fwds, {s_sp['mean_tok_per_fwd']:.2f} tok/fwd, "
          f"accept={s_sp['accept_per_offset']}, "
          f"equal_ar={sp_text == ar_text}")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[spec] -> {a.out}")


if __name__ == "__main__":
    main()
