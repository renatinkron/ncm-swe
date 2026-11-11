#!/usr/bin/env python3
"""Train the learned last mile: token classifiers on frozen
backbone+pyramid heads. TWO variants share one backbone forward:

  feat   — f(feat) -> V logits            (does feat alone suffice?)
  codes  — f(feat, predicted code vecs)   (do RVQ codes add signal?)

The codes variant consumes the codes the frozen LevelHeads actually
predict (greedy per level, sequential conditioning) — matching what is
available at inference, not teacher-forced truth.

Target = the true next token (real supervision). Logged metrics every
`--eval-every` steps on a held-out batch: loss + top-1 vs truth + top-1
vs the backbone's own AR argmax (feat @ E.T — the spec-decode verify).

Usage:
  python scripts/train_token_head.py --model-dir ~/models/qwen05-sft \
      --steps 1500 --out artifacts/token_head_qwen05 --resume
"""
import argparse, json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_level_heads import FrozenQwen, LevelHeads  # noqa: E402


class TokenHead(nn.Module):
    def __init__(self, d_in, V):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.SiLU(),
                                 nn.Linear(512, V))


@torch.no_grad()
def predicted_code_vecs(heads, feat, cbs):
    """Greedy codes per level (sequential) → concat code vectors. B×(L*d)"""
    h = heads.proj(feat)
    prev_vecs, out_vecs = [], []
    for k in range(heads.L):
        if k == 0:
            logits = heads.out[0](h)
        else:
            ctx = torch.cat(prev_vecs, -1)
            logits = heads.out[k](heads.cond[k](ctx) + h)
        c = logits.argmax(-1)
        v = cbs[k][c]
        prev_vecs.append(v)
        out_vecs.append(v)
    return torch.cat(out_vecs, -1)


def accs(logits, truth, ar_tok):
    t1 = int((logits.argmax(-1) == truth).sum())
    a1 = int((logits.argmax(-1) == ar_tok).sum())
    return t1, a1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--codebook", default="artifacts/pyramid_cb_qwen05.pt")
    ap.add_argument("--heads", default="artifacts/level_heads_qwen05/final.pt")
    ap.add_argument("--stream", default="artifacts/qwen_stream.pt")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default="artifacts/token_head_qwen05")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target", choices=["truth", "ar"], default="truth",
                    help="truth: corpus token (hard CE); ar: frozen AR "
                         "logits at same position (soft KL, DistillSpec)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="softmax temperature for --target ar")
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
    heads.load_state_dict(torch.load(a.heads, weights_only=False)["model"])
    for p in heads.parameters():
        p.requires_grad_(False)
    heads.eval()

    variants = nn.ModuleDict({
        "feat": TokenHead(d, V),
        "codes": TokenHead(d + L * d, V),
    })
    npar = {k: sum(p.numel() for p in v.parameters()) for k, v in
            variants.items()}
    print(f"[th] params: {npar}")

    toks = torch.load(a.stream, weights_only=False)
    toks = toks[:toks.numel() // a.seq * a.seq].view(-1, a.seq)
    n_hold = int(toks.shape[0] * 0.05)
    train_pool, hold = toks[:-n_hold], toks[-n_hold:]

    opt = torch.optim.AdamW(variants.parameters(), lr=a.lr)
    start = 0
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, weights_only=False)
        variants.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start = sd["step"]
        print(f"[th] resumed at {start}")

    os.makedirs(a.out, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    hist = {"feat": [], "codes": []}
    for step in range(start, a.steps):
        is_eval = step % a.eval_every == 0
        split = hold if is_eval else train_pool
        idx = torch.randint(0, split.shape[0], (a.batch,), generator=g)
        ids = split[idx]
        inp, tgt = ids[:, :-1], ids[:, -1]
        feat = model(inp)[:, -1]
        ar_logits = feat @ E.t()
        ar_tok = ar_logits.argmax(-1)
        cvecs = predicted_code_vecs(heads, feat, cbs)
        if a.target == "ar":
            soft_tgt = F.log_softmax(ar_logits / a.temperature, -1)
        for name, th in variants.items():
            x = feat if name == "feat" else torch.cat([feat, cvecs], -1)
            logits = th.net(x)
            if a.target == "ar":
                loss = -(soft_tgt.exp() * F.log_softmax(logits, -1)).sum(-1).mean()
            else:
                loss = F.cross_entropy(logits, tgt)
            if is_eval:
                t1, a1 = accs(logits, tgt, ar_tok)
                hist[name].append({"step": step, "loss": float(loss),
                                   "t1": t1 / a.batch, "a1": a1 / a.batch})
                print(f"[th:{name}] eval@{step} loss={float(loss):.3f} "
                      f"top1_truth={t1}/{a.batch} top1_ar={a1}/{a.batch}",
                      flush=True)
            else:
                opt.zero_grad()
                loss.backward()
        if not is_eval:
            opt.step()
        if step % 50 == 0:
            print(f"[th] step {step}/{a.steps} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % 250 == 0:
            torch.save({"model": variants.state_dict(),
                        "opt": opt.state_dict(), "step": step + 1},
                       ckpt_path)
    torch.save({"model": variants.state_dict(), "step": a.steps,
                "hist": hist}, os.path.join(a.out, "final.pt"))
    print(f"[th] done in {time.time()-t0:.0f}s -> {a.out}/final.pt")


if __name__ == "__main__":
    main()
