"""RUN-028: decode pipeline stage timing (gen / vqvae / pick_tokens / CE)."""
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
def l0_dist(mode, path, vocab):
    if mode == "model":
        return None
    p = np.load(path, allow_pickle=False).astype(np.float64)
    p = torch.tensor(p / p.sum())
    assert p.numel() == vocab, f"unigram {p.numel()} != vocab {vocab}"
    return p.float()


def pick_tokens(logits, freq=0.0, mu=0.0, adj=None, allow=None):
    if freq <= 0 and not adj:
        return logits.argmax(-1)
    B, T, V = logits.shape
    lg = logits.double()
    alw = (allow[:V] * T)[None, :] if allow is not None else None
    al = adj[0] if adj else 0.0
    aw = int(adj[1]) if adj and len(adj) > 1 else 1
    ad = adj[2] if adj and len(adj) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            excess = (cnt - alw).clamp_min(0.0) if alw is not None else cnt
            step -= freq * excess
            if mu > 0 and alw is not None:
                step += mu * ((T - t) / T) * (alw - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None],
                              torch.full((B, 1), -al * ad ** j,
                                         dtype=torch.float64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64,
                                                   device=lg.device))
    return out
CK = "/work/artifacts/ncm_official"
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
res = {}
with torch.no_grad():
    p = l0_dist("unigram", "/work/artifacts/l0_unigram.npy",
                int(model.vocab_sizes[0]))
    L0 = model.levels[0]
    l0c = torch.multinomial(p, 8 * L0, replacement=True).view(8, L0)
    t = time.perf_counter()
    idx = model.generate(batch_size=8, temperature=1.0, top_k=0,
                         top_p=1.0, l0=l0c)
    res["generate_B8_s"] = round(time.perf_counter() - t, 2)
    res["passes"] = model.last_passes
    t = time.perf_counter()
    _, zl = vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
    logits = vqvae.decoder(zl[-1])
    res["vqvae_decode_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    ids = pick_tokens(logits)
    res["pick_tokens_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    model.corrupt = 0.0
    sl, _ = model.loss(idx)
    held = np.load("/work/artifacts/codes_official/held.npy").astype(np.int64)
    bi = np.random.randint(0, len(held), 8)
    gl, _ = model.loss(torch.tensor(held[bi], dtype=torch.long))
    res["ce_eval_2xfwd_s"] = round(time.perf_counter() - t, 2)
print(json.dumps(res, indent=1))
