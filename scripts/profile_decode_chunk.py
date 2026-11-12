"""RUN-028: verify chunk-local decode bit-identity + measure speedup.

Hypothesis: with block_diagonal attention, the last chunk's outputs depend
only on [q0 + own chunk] — earlier code chunks are masked out. The official
upstream generate() exploits this (only current chunk passes the trunk).
Test: full-prefix _forward_partial logits vs chunk-local forward logits.
"""
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
torch.manual_seed(0)

CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vqvae = tk.vqvae.eval()
for p in vqvae.parameters(): p.requires_grad_(False)
model = LevelGenerator(vqvae, n_embd=896, n_layer=4, n_head_q=14,
                       n_head_kv=2, mlp_hidden=4864, intra=True,
                       intra_mask="mix", corrupt=0.0)
ck = torch.load("/work/artifacts/gen_ncm_v5/final.pt", map_location="cpu",
                weights_only=False)
model.load_state_dict(ck["model"], strict=True)
model.eval()

held = np.load(f"{CODES}/held.npy").astype(np.int64)
B = 4
idx = torch.tensor(held[:B], dtype=torch.long)

res = {}
with torch.no_grad():
    for k in [2, 5, 8, 11]:
        # codes for levels 0..k-1 from ground truth
        ofs = 0
        codes = []
        for L in model.levels[:k]:
            codes.append(idx[:, ofs:ofs + L])
            ofs += L
        # full-prefix forward (current impl)
        h_full = model._forward_partial(codes)
        Lk = model.levels[k]
        lg_full = model._head_logits(h_full[:, -Lk:], k)

        # chunk-local forward: [q0 + chunk_k] with GLOBAL pos/rope/lvl ids
        chunks = model._chunks_from_codes(codes, k)
        ch = chunks[-1]
        gofs = 1 + sum(model.levels[1:k])  # global offset of chunk k in seq
        L = ch.shape[1]
        h0 = model.q0.expand(B, -1, -1) + model.pos.weight[0] \
            + model.lvl.weight[0]
        hc = model.proj_in(ch) + model.pos.weight[gofs:gofs + L] \
            + model.lvl.weight[k]
        hloc = torch.cat([h0, hc], dim=1)
        # mask: q0 sees itself only; chunk sees all
        m = torch.zeros(1, 1, 1 + L, 1 + L)
        m[0, 0, 0, 1:] = float("-inf")
        # rope at GLOBAL positions: [0] + gofs..gofs+L-1
        pos_ids = torch.cat([torch.zeros(1, dtype=torch.long),
                             torch.arange(gofs, gofs + L)])
        cos = model.rope_cos[pos_ids]
        sin = model.rope_sin[pos_ids]
        for blk in model.blocks:
            hloc, _ = blk(hloc, m, cos, sin)
        hloc = model.ln_f(hloc)
        lg_loc = model._head_logits(hloc[:, 1:], k)

        d = (lg_full - lg_loc).abs()
        res[f"k{k}"] = {"max_abs_diff": float(d.max()),
                        "mean_abs_diff": float(d.mean()),
                        "argmax_match": float(
                            (lg_full.argmax(-1) == lg_loc.argmax(-1)
                             ).float().mean())}

    # timing: full _forward_partial vs chunk-local at k=11, B=1 and B=8
    for Bb in [1, 8]:
        ofs = 0
        codes = []
        for L in model.levels[:10]:
            codes.append(torch.randint(0, 512, (Bb, L)))
        Lk = model.levels[10]
        for _ in range(3): model._forward_partial(codes)
        t = time.perf_counter()
        for _ in range(8): model._forward_partial(codes)
        full_ms = (time.perf_counter() - t) / 8 * 1000

        chunks = model._chunks_from_codes(codes, 10)
        ch = chunks[-1]
        gofs = 1 + sum(model.levels[1:10])
        L = ch.shape[1]
        m = torch.zeros(1, 1, 1 + L, 1 + L)
        m[0, 0, 0, 1:] = float("-inf")
        pos_ids = torch.cat([torch.zeros(1, dtype=torch.long),
                             torch.arange(gofs, gofs + L)])
        cos = model.rope_cos[pos_ids]; sin = model.rope_sin[pos_ids]
        def loc():
            h0 = model.q0.expand(Bb, -1, -1) + model.pos.weight[0] \
                + model.lvl.weight[0]
            hc = model.proj_in(ch) + model.pos.weight[gofs:gofs + L] \
                + model.lvl.weight[10]
            hloc = torch.cat([h0, hc], 1)
            for blk in model.blocks:
                hloc, _ = blk(hloc, m, cos, sin)
            return model.ln_f(hloc)
        for _ in range(3): loc()
        t = time.perf_counter()
        for _ in range(8): loc()
        loc_ms = (time.perf_counter() - t) / 8 * 1000
        res[f"time_B{Bb}_k10"] = {"full_prefix_ms": round(full_ms, 1),
                                 "chunk_local_ms": round(loc_ms, 1),
                                 "speedup": round(full_ms / loc_ms, 2)}

    # projected whole-decode speedup: sum over levels of passes*(positions)
    lv = model.levels
    for ip in [32]:
        cur_tot = sum(
            ((L + ip - 1) // ip) * (1 + sum(lv[1:i + 1]))
            for i, L in enumerate(lv[1:], start=1))
        loc_tot = sum(
            ((L + ip - 1) // ip) * (1 + L) for L in lv[1:])
        res["positions_total"] = {"current": cur_tot,
                                  "chunk_local": loc_tot,
                                  "ratio": round(cur_tot / loc_tot, 2)}

json.dump(res, open("/repo/artifacts/prof/decode_chunk.json", "w"), indent=2)
print(json.dumps(res, indent=1))
