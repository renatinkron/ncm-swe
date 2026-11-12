"""Kernel-level A/B for RUN-027: gqa, rms_norm, per-level attn, batch."""
import importlib, json, os, sys, time, types
hydra = types.ModuleType("hydra")
class _U:
    @staticmethod
    def get_class(p):
        m, _, n = p.rpartition(".")
        return getattr(importlib.import_module(m), n)
hydra.utils = _U
sys.modules["hydra"] = hydra
ROOT = "/repo/vendor/ncm-upstream"
sys.path.insert(0, ROOT); sys.path.insert(0, "/repo/src")
os.chdir(ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from ncm_swe.model.generator import LevelGenerator
from ncm_swe.model.common import RMSNorm
CK = "/work/artifacts/ncm_official"
CODES = "/work/artifacts/codes_official"
torch.set_num_threads(8)
res = {}

# 1) attention: dense-mask vs enable_gqa vs per-level (fwd+bwd)
B, HQ, HK, D, T = 16, 14, 2, 64, 641
q = torch.randn(B, HQ, T, D, requires_grad=True)
k2 = torch.randn(B, HK, T, D, requires_grad=True)
v2 = torch.randn(B, HK, T, D, requires_grad=True)
mask = torch.full((1, 1, T, T), float("-inf"))
torch.manual_seed(0)
import random
lvl_sizes = [8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128]
ofs = 1
for L in lvl_sizes:
    mask[0, 0, ofs:ofs + L, ofs:ofs + L] = 0.0
    ofs += L
mask[0, 0, 1:, 0] = 0.0
mask[0, 0, 0, 0] = 0.0

def bench(fn, n=5):
    for _ in range(2): fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return round((time.perf_counter() - t) / n * 1000, 2)

def dense_mask():
    k = k2.repeat_interleave(HQ // HK, dim=1)
    v = v2.repeat_interleave(HQ // HK, dim=1)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    y.sum().backward()
def gqa():
    y = F.scaled_dot_product_attention(q, k2, v2, attn_mask=mask,
                                       enable_gqa=True)
    y.sum().backward()
def per_level():
    outs = []
    o = 0
    qs = torch.cat([q[:, :, :1], q[:, :, 1:]], 2)  # q0 attends self only -> handle separately
    for L in lvl_sizes:
        kk = torch.cat([k2[:, :, :1], k2[:, :, o:o + L]], dim=2)
        vv = torch.cat([v2[:, :, :1], v2[:, :, o:o + L]], dim=2)
        qq = q[:, :, 1 + o:1 + o + L]
        outs.append(F.scaled_dot_product_attention(
            qq, kk, vv, enable_gqa=True))
        o += L
    outs.append(F.scaled_dot_product_attention(
        q[:, :, :1], k2[:, :, :1], v2[:, :, :1], enable_gqa=True))
    torch.cat(outs, 2).sum().backward()

res["attn_dense_mask_fwdbwd_ms"] = bench(dense_mask)
res["attn_gqa_fwdbwd_ms"] = bench(gqa)
res["attn_per_level_fwdbwd_ms"] = bench(per_level)

# 2) RMSNorm python vs F.rms_norm (fwd+bwd)
x = torch.randn(B, T, 896, requires_grad=True)
w = torch.ones(896, requires_grad=True)
rn = RMSNorm(896)
def py_rms():
    y = rn(x); y.sum().backward()
def fused_rms():
    y = F.rms_norm(x, (896,), w, 1e-6); y.sum().backward()
res["rmsnorm_py_fwdbwd_ms"] = bench(py_rms)
res["rmsnorm_fused_fwdbwd_ms"] = bench(fused_rms)

# 3) full step batch 16 vs 32
CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vqvae = tk.vqvae.eval()
for p in vqvae.parameters(): p.requires_grad_(False)
model = LevelGenerator(vqvae, n_embd=896, n_layer=4, n_head_q=14,
                       n_head_kv=2, mlp_hidden=4864, intra=True,
                       intra_mask="mix", corrupt=1.0,
                       corrupt_schedule="linear", corrupt_hi=0.75,
                       corrupt_lo=0.25)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
tr = np.load(f"{CODES}/train.npy").astype(np.int64)
for bs in [16, 32]:
    ts = []
    for s in range(4):
        bi = np.random.randint(0, len(tr), bs)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        t = time.perf_counter()
        l, _ = model.loss(idx)
        opt.zero_grad(set_to_none=True); l.backward(); opt.step()
        ts.append(time.perf_counter() - t)
    res[f"step_b{bs}_s"] = round(sorted(ts)[len(ts)//2], 3)
    res[f"per_sample_b{bs}_ms"] = round(sorted(ts)[len(ts)//2] / bs * 1000, 1)

json.dump(res, open("/repo/artifacts/prof/kernel_alts.json", "w"), indent=2)
print(json.dumps(res, indent=1))
