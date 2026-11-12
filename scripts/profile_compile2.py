"""Isolate torch.compile failure: try single Block / trunk-only (RUN-027)."""
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
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from ncm_swe.model.generator import LevelGenerator
CK = "/work/artifacts/ncm_official"
CODES = "/work/artifacts/codes_official"
torch.set_num_threads(8)
res = {}

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

# 1) single block
blk = model.blocks[0]
x = torch.randn(16, 641, 896, requires_grad=True)
mask = model.attn_mask
cos, sin = model.rope_cos, model.rope_sin
try:
    cblk = torch.compile(blk)
    for _ in range(3):
        y, _ = cblk(x, mask, cos, sin)
        y.sum().backward()
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        y, _ = cblk(x, mask, cos, sin); y.sum().backward()
        ts.append(time.perf_counter() - t)
    res["block_compiled_ms"] = round(sorted(ts)[2] * 1000, 1)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        y, _ = blk(x, mask, cos, sin); y.sum().backward()
        ts.append(time.perf_counter() - t)
    res["block_eager_ms"] = round(sorted(ts)[2] * 1000, 1)
except Exception as e:
    res["block_compile_error"] = repr(e)[:200]

# 2) trunk-only wrapper
def trunk(h):
    for b in model.blocks:
        h, _ = b(h, mask, cos, sin)
    return h
try:
    ctrunk = torch.compile(trunk)
    for _ in range(3):
        ctrunk(x.detach().requires_grad_(True)).sum().backward()
    ts = []
    for _ in range(5):
        xin = x.detach().requires_grad_(True)
        t = time.perf_counter()
        ctrunk(xin).sum().backward()
        ts.append(time.perf_counter() - t)
    res["trunk_compiled_ms"] = round(sorted(ts)[2] * 1000, 1)
    ts = []
    for _ in range(5):
        xin = x.detach().requires_grad_(True)
        t = time.perf_counter()
        trunk(xin).sum().backward()
        ts.append(time.perf_counter() - t)
    res["trunk_eager_ms"] = round(sorted(ts)[2] * 1000, 1)
except Exception as e:
    res["trunk_compile_error"] = repr(e)[:200]

json.dump(res, open("/repo/artifacts/prof/compile2.json", "w"), indent=2)
print(json.dumps(res, indent=1))
