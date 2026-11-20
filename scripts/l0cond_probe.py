"""l0cond probe: does a frozen-LLM feature predict the segment's l0 code?

Trains {linear, mlp} classifiers feature(896) -> logits(4096) on cached
features from scripts/l0cond_features.py. Reports held-out CE / top-1 /
top-8 against three baselines:

  uniform  — ln(4096)
  marginal — train-split l0 distribution applied to held (the "unigram"
             floor: what --l0 unigram sampling already achieves)
  shuffle  — same pipeline, labels permuted (leakage control; must not
             beat marginal)

  ~/.venvs/ncm/bin/python scripts/l0cond_probe.py [--steps 3000]
"""
import argparse, json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = "artifacts/l0cond"
FEATS = {"E": ("feat_E", "l0"), "H": ("feat_H", "l0")}


def make_head(arch, d, v):
    if arch == "linear":
        return nn.Linear(d, v)
    return nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, v))


def fit_eval(Xtr, ytr, Xho, yho, arch, steps, bs, lr, seed):
    torch.manual_seed(seed)
    d, v = Xtr.shape[1], 4096
    net = make_head(arch, d, v)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    n = Xtr.shape[0]
    t0 = time.time()
    for s in range(steps):
        idx = torch.randint(0, n, (bs,))
        loss = F.cross_entropy(net(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        ce_sum, t1, t8 = 0.0, 0, 0
        B = 4096
        for i in range(0, Xho.shape[0], B):
            lg = net(Xho[i:i + B])
            ce_sum += F.cross_entropy(
                lg, yho[i:i + B], reduction="sum").item()
            t1 += (lg.argmax(-1) == yho[i:i + B]).sum().item()
            t8 += (lg.topk(8, -1).indices
                   == yho[i:i + B].unsqueeze(-1)).any(-1).sum().item()
    m = Xho.shape[0]
    return {"ce": ce_sum / m, "top1": t1 / m, "top8": t8 / m,
            "train_s": round(time.time() - t0, 1), "net": net}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "10")))

    ytr_all = np.load(f"{OUT}/l0_train.npy")
    yho_all = np.load(f"{OUT}/l0_held.npy")

    # marginal baseline: train l0 histogram -> held CE (unigram floor).
    # Add-alpha smoothing so unseen held codes stay finite; also report
    # coverage (fraction of held codes seen in train). Computed over the
    # same eval slice each arm uses (per-arm values below).
    alpha = 1.0
    hist = np.bincount(ytr_all, minlength=4096).astype(np.float64)
    p = (hist + alpha) / (hist.sum() + alpha * 4096)
    held_ce_marginal = float(-np.log(p[yho_all]).mean())
    coverage = float((hist[yho_all] > 0).mean())

    res = {"baselines": {
        "uniform": math.log(4096),
        "marginal": held_ce_marginal,
        "marginal_alpha": alpha,
        "l0_coverage_held": coverage,
        "held_n": int(yho_all.shape[0]),
        "l0_vocab": 4096}}

    for arm, (fpat, lpat) in FEATS.items():
        Xtr = np.load(f"{OUT}/{fpat}_train.npy").astype(np.float32)
        Xho = np.load(f"{OUT}/{fpat}_held.npy").astype(np.float32)
        ytr, yho = torch.tensor(ytr_all[:Xtr.shape[0]]), \
            torch.tensor(yho_all[:Xho.shape[0]])
        Xtr, Xho = torch.from_numpy(Xtr), torch.from_numpy(Xho)
        n_eval = min(args.eval_n, Xho.shape[0])
        Xe, ye = Xho[:n_eval], yho[:n_eval]
        # per-arm marginal on the SAME eval slice the heads see
        res[f"{arm}_marginal_slice"] = float(
            -np.log(p[ye.numpy()]).mean())
        best_ce = 1e9
        for arch in ["linear", "mlp"]:
            r = fit_eval(Xtr, ytr, Xe, ye, arch, args.steps,
                         args.batch, args.lr, args.seed)
            net = r.pop("net")
            res[f"{arm}_{arch}"] = r
            if r["ce"] < best_ce:
                best_ce = r["ce"]
                torch.save({"sd": net.state_dict(), "arch": arch,
                            "feature": arm, "d": 896, "v": 4096},
                           f"{OUT}/adapter_{arm}.pt")
            print(f"{arm}_{arch}: ce={r['ce']:.3f} t1={r['top1']:.3f} "
                  f"t8={r['top8']:.3f} ({r['train_s']}s)", flush=True)
        # shuffle control on the linear arm
        g = torch.Generator().manual_seed(args.seed)
        yperm = ytr[torch.randperm(ytr.shape[0], generator=g)]
        r = fit_eval(Xtr, yperm, Xe, ye, "linear", args.steps,
                     args.batch, args.lr, args.seed + 1)
        r.pop("net")
        res[f"{arm}_shuffle_linear"] = r
        print(f"{arm}_shuffle: ce={r['ce']:.3f} t1={r['top1']:.3f}",
              flush=True)

    json.dump(res, open(f"{OUT}/probe.json", "w"), indent=2)
    print("-> artifacts/l0cond/probe.json")


if __name__ == "__main__":
    main()
