"""Profile training step of LevelGenerator (RUN-027 research).

Replicates train_gen_official.py env exactly (v5 arm: intra=mix,
linear corruption), then measures:
  A) phase timing: input-prep / blocks / head+loss / backward / opt
  B) torch.profiler op-level table over N steps
  C) roofline microbenchmarks: GEMM GFLOPS, SDPA variants, thread sweep
"""
import argparse, importlib, json, os, sys, time, types

hydra = types.ModuleType("hydra")
class _U:
    @staticmethod
    def get_class(p):
        m, _, n = p.rpartition(".")
        return getattr(importlib.import_module(m), n)
hydra.utils = _U
sys.modules["hydra"] = hydra

ROOT = "/repo/vendor/ncm-upstream"
sys.path.insert(0, ROOT)
sys.path.insert(0, "/repo/src")
os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
import numpy as np
import torch
import torch.nn.functional as F
from ncm_swe.model.generator import LevelGenerator

CK = "/work/artifacts/ncm_official"
CODES = "/work/artifacts/codes_official"

def build(threads, ckpt="/work/artifacts/gen_ncm_v5/final.pt",
          intra=True, intra_mask="mix"):
    torch.set_num_threads(threads)
    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vqvae = tk.vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    m = LevelGenerator(vqvae, n_embd=896, n_layer=4, n_head_q=14,
                       n_head_kv=2, mlp_hidden=4864,
                       intra=intra, intra_mask=intra_mask,
                       corrupt=1.0, corrupt_schedule="linear",
                       corrupt_hi=0.75, corrupt_lo=0.25)
    if ckpt and os.path.exists(ckpt):
        prev = torch.load(ckpt, map_location="cpu", weights_only=False)
        m.load_state_dict(prev["model"], strict=False)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ckpt", default="/work/artifacts/gen_ncm_v5/final.pt")
    ap.add_argument("--out", default="/repo/artifacts/prof/profile.json")
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    model = build(args.threads, args.ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.95), weight_decay=0.1)
    tr = np.load(f"{CODES}/train.npy").astype(np.int64)

    # ---- A) phase timing -------------------------------------------------
    # instrument forward internals via hooks on the heavy pieces
    ph = {"data": 0.0, "prep": 0.0, "blocks": 0.0, "head_loss": 0.0,
          "bwd": 0.0, "opt": 0.0}
    t_hold = [0.0]
    def pre_blocks(mod, inp): t_hold[0] = time.perf_counter()
    def post_head(mod, inp, out): ph["blocks"] += time.perf_counter() - t_hold[0]
    model.blocks[-1].register_forward_pre_hook(pre_blocks)
    model.blocks[-1].register_forward_hook(post_head)
    orig_fwd = model.forward
    def fwd_timed(idx, boundaries=None, decided=None):
        t = time.perf_counter()
        r = orig_fwd(idx, boundaries, decided)
        ph["prep"] += (time.perf_counter() - t) - ph["_b"]
        return r
    # blocks time is captured by hooks; prep = fwd_total - blocks - head_slicing
    for step in range(args.steps):
        ph["_b"] = 0.0
        t = time.perf_counter()
        bi = np.random.randint(0, len(tr), args.batch)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        ph["data"] += time.perf_counter() - t
        t = time.perf_counter()
        boundaries, decided = model._sample_intra_masks(idx)
        logits_l0, outs = orig_fwd(idx, boundaries, decided)
        t_fwd = time.perf_counter() - t
        ph["prep"] += t_fwd - ph["blocks"]  # blocks accum inside
        t = time.perf_counter()
        l0 = model.levels[0]
        loss = F.cross_entropy(
            logits_l0.reshape(-1, logits_l0.shape[-1]),
            idx[:, :l0].reshape(-1))
        ofs = l0
        for i, out in enumerate(outs):
            L = model.chunk_levels[i]
            t_ = idx[:, ofs:ofs + L]
            pending = ~decided[i] if decided is not None else (
                torch.arange(L) >= boundaries[:, i:i + 1])
            ce = (F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                  t_.reshape(-1), reduction="none")
                  .reshape(t_.shape) * pending).sum() / pending.sum().clamp_min(1)
            loss = loss + ce
            ofs += L
        loss = loss / (len(outs) + 1)
        ph["head_loss"] += time.perf_counter() - t
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        ph["bwd"] += time.perf_counter() - t
        t = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ph["opt"] += time.perf_counter() - t
    ph.pop("_b", None)
    n = args.steps
    res = {"phase_per_step_s": {k: round(v / n, 4) for k, v in ph.items()},
           "threads": args.threads, "batch": args.batch,
           "steps": n}

    # ---- B) torch.profiler op table -------------------------------------
    from torch.profiler import profile, ProfilerActivity
    for _ in range(2):  # warmup
        bi = np.random.randint(0, len(tr), args.batch)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        l, _ = model.loss(idx); opt.zero_grad(set_to_none=True); l.backward()
        opt.step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(4):
            bi = np.random.randint(0, len(tr), args.batch)
            idx = torch.tensor(tr[bi], dtype=torch.long)
            l, _ = model.loss(idx)
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    rows = []
    for e in prof.key_averages():
        rows.append({"op": e.key,
                     "self_ms": round(e.self_cpu_time_total / 1000 / 4, 1),
                     "tot_ms": round(e.cpu_time_total / 1000 / 4, 1),
                     "calls": e.count // 4})
    rows.sort(key=lambda r: -r["self_ms"])
    res["ops_per_step"] = rows[:25]

    # ---- C) microbenchmarks ----------------------------------------------
    mb = {}
    BT = args.batch * (model.T + 1)
    # trunk GEMMs: qkv+o (per layer) and mlp (3 GEMMs)
    def bench_mm(m, k, n_, iters=6):
        a = torch.randn(m, k); b = torch.randn(k, n_)
        for _ in range(2): a @ b
        t = time.perf_counter()
        for _ in range(iters): a @ b
        dt = (time.perf_counter() - t) / iters
        return round(2 * m * k * n_ / dt / 1e9, 1), round(dt * 1000, 2)
    mb["gemm_10256x896x4864_gflops,ms"] = bench_mm(BT, 896, 4864)
    mb["gemm_10256x4864x896_gflops,ms"] = bench_mm(BT, 4864, 896)
    mb["gemm_10256x896x1152_gflops,ms"] = bench_mm(BT, 896, 1152)
    # SDPA block-diagonal dense vs per-level chunked vs unmasked
    q = torch.randn(args.batch, 14, 641, 64)
    k = torch.randn(args.batch, 14, 641, 64)
    v = torch.randn(args.batch, 14, 641, 64)
    with torch.no_grad():
        for _ in range(2):
            F.scaled_dot_product_attention(q, k, v, attn_mask=model.attn_mask)
        t = time.perf_counter()
        for _ in range(6):
            F.scaled_dot_product_attention(q, k, v, attn_mask=model.attn_mask)
        mb["sdpa_dense_mask_ms"] = round((time.perf_counter() - t) / 6 * 1000, 2)
        # per-level: separate sdpa per chunk (what sparsity-aware would do)
        chunks = [1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
        t = time.perf_counter()
        for _ in range(6):
            ofs = 0
            for L in chunks:
                F.scaled_dot_product_attention(
                    q[:, :, ofs:ofs + L], k[:, :, ofs:ofs + L],
                    v[:, :, ofs:ofs + L])
                ofs += L
        mb["sdpa_per_level_ms"] = round((time.perf_counter() - t) / 6 * 1000, 2)
    res["microbench"] = mb
    res["model"] = {"T": model.T, "params_M": round(
        sum(p.numel() for p in model.parameters()) / 1e6, 1)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=1))

main()
