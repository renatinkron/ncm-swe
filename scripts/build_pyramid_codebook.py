#!/usr/bin/env python3
"""Build a residual-VQ 'pyramid' codebook directly from a pretrained LLM's
embedding matrix — no VQ-VAE training (the 'palette from a mastered
painting' trick: codebook derived from an existing semantic space).

Each token embedding e is decomposed into L levels:
  e ≈ c_0[i0] + c_1[i1] + ... + c_{L-1}[i_{L-1}]      (residual per level)

Level 0 = coarse semantic region ('average colour of the region'),
final level = exact token. Metrics reported: recon MSE per level,
alive codes per level, and top-1 token recovery (nearest-neighbour over
the full vocab of the reconstructed vector returns the right token).

Usage:
  python scripts/build_pyramid_codebook.py --model-dir /tmp/llm/qwen15 \
      --levels 128,256,1024,4096,8192 --out artifacts/pyramid_cb.pt
"""
import argparse, json, os, time
import torch
from safetensors.torch import load_file


def load_embeddings(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    st = load_file(os.path.join(model_dir, "model.safetensors"))
    key = "model.embed_tokens.weight"
    if key not in st:
        key = next(k for k in st if "embed_tokens.weight" in k)
    emb = st[key].float()
    print(f"[pyr] embeddings {tuple(emb.shape)} from {key} "
          f"(tie={cfg.get('tie_word_embeddings')})")
    return emb


def nearest_chunked(X, C, block=4096):
    """argmin ||x - c|| row-wise, chunked to bound memory."""
    out = torch.empty(X.shape[0], dtype=torch.long)
    cn = C.pow(2).sum(1)
    for s in range(0, X.shape[0], block):
        xb = X[s:s + block]
        d = xb.pow(2).sum(1, keepdim=True) + cn - 2 * xb @ C.t()
        out[s:s + block] = d.argmin(1)
    return out


def train_level(residual, K, iters=25, seed=0):
    """EMA/kmeans hybrid: seed from data, iterate assign+mean (bounded)."""
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(residual.shape[0], generator=g)[:K]
    C = residual[sel].clone()
    for _ in range(iters):
        idx = nearest_chunked(residual, C)
        for k in range(K):
            m = idx == k
            if m.any():
                C[k] = residual[m].mean(0)
    return C, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--levels", default="128,256,1024,4096,8192")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--out", default="artifacts/pyramid_cb.pt")
    ap.add_argument("--subsample", type=int, default=0,
                    help="quantize only first N tokens (0 = all)")
    a = ap.parse_args()

    E = load_embeddings(a.model_dir)
    V, d = E.shape
    if a.subsample:
        E = E[:a.subsample]; V = a.subsample
    sizes = [int(x) for x in a.levels.split(",")]
    K = len(sizes)

    t0 = time.time()
    residual = E.clone()
    codebooks, all_idx, report = [], [], {"levels": []}
    print(f"[pyr] V={V} d={d} levels={sizes}")

    for li, sz in enumerate(sizes):
        C, idx = train_level(residual, sz, iters=a.iters, seed=li)
        q = C[idx]
        mse = float((E - (E - residual + q)).pow(2).mean())
        recon = E - residual + q            # cumulative reconstruction
        cum_mse = float((E - recon).pow(2).mean())
        alive = int(idx.unique().numel())
        residual = residual - q
        codebooks.append(C); all_idx.append(idx)
        row = {"level": li, "codes": sz, "alive": alive,
               "cum_mse": round(cum_mse, 6),
               "residual_mse": round(float(residual.pow(2).mean()), 6)}
        report["levels"].append(row)
        print(f"[pyr] L{li} codes={sz} alive={alive} "
              f"cum_mse={cum_mse:.5f} res={row['residual_mse']:.5f}")

    # ---- top-1 token recovery (chunked NN over full vocab) ----------
    recon = E - residual
    correct = 0
    for s in range(0, V, 2048):
        xb = recon[s:s + 2048]
        D = xb.pow(2).sum(1, keepdim=True) + E.pow(2).sum(1) - 2 * xb @ E.t()
        correct += int((D.argmin(1) == torch.arange(s, s + xb.shape[0])).sum())
    top1 = correct / V
    report["top1_recovery"] = round(top1, 4)
    report["V"] = V; report["d"] = d
    report["wall_s"] = round(time.time() - t0, 1)
    print(f"[pyr] top-1 token recovery: {top1*100:.1f}%  ({correct}/{V})")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"codebooks": codebooks, "levels": sizes,
                "embed_dim": d, "report": report,
                "model_dir": a.model_dir}, a.out)
    json.dump(report, open(a.out.replace(".pt", "_report.json"), "w"), indent=2)
    print(f"[pyr] saved {a.out} in {report['wall_s']}s")


if __name__ == "__main__":
    main()
