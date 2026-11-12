"""Quick A/B: thread sweep + torch.compile on loss step (RUN-027)."""
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
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
import numpy as np
import torch
from ncm_swe.model.generator import LevelGenerator
CK = "/work/artifacts/ncm_official"
CODES = "/work/artifacts/codes_official"

def build():
    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vqvae = tk.vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    return LevelGenerator(vqvae, n_embd=896, n_layer=4, n_head_q=14,
                          n_head_kv=2, mlp_hidden=4864, intra=True,
                          intra_mask="mix", corrupt=1.0,
                          corrupt_schedule="linear",
                          corrupt_hi=0.75, corrupt_lo=0.25)

tr = np.load(f"{CODES}/train.npy").astype(np.int64)
res = {}
model = build()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for th in [8, 16]:
    torch.set_num_threads(th)
    ts = []
    for s in range(6):
        bi = np.random.randint(0, len(tr), 16)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        t = time.perf_counter()
        l, _ = model.loss(idx)
        opt.zero_grad(set_to_none=True); l.backward(); opt.step()
        ts.append(time.perf_counter() - t)
    res[f"step_threads{th}"] = round(sorted(ts)[len(ts)//2], 3)

# torch.compile on forward (loss keeps python loop over levels)
torch.set_num_threads(8)
try:
    t0 = time.time()
    cfwd = torch.compile(model.forward)
    for s in range(3):  # warmup+compile
        bi = np.random.randint(0, len(tr), 16)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        b_, d_ = model._sample_intra_masks(idx)
        cfwd(idx, b_, d_)
    res["compile_warmup_s"] = round(time.time() - t0, 1)
    ts = []
    for s in range(6):
        bi = np.random.randint(0, len(tr), 16)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        t = time.perf_counter()
        b_, d_ = model._sample_intra_masks(idx)
        logits_l0, outs = cfwd(idx, b_, d_)
        l0 = model.levels[0]
        loss = torch.nn.functional.cross_entropy(
            logits_l0.reshape(-1, logits_l0.shape[-1]),
            idx[:, :l0].reshape(-1))
        for o in outs:
            loss = loss + o.sum() * 0  # proxy: keep graph alive cheaply
        loss.backward()
        opt.zero_grad(set_to_none=True)
        ts.append(time.perf_counter() - t)
    res["step_compiled_fwd"] = round(sorted(ts)[len(ts)//2], 3)
except Exception as e:
    res["compile_error"] = repr(e)[:300]

json.dump(res, open("/repo/artifacts/prof/variants.json", "w"), indent=2)
print(json.dumps(res, indent=1))
