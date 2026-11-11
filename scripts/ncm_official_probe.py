"""Probe the official NCM TinyStories checkpoint on CPU.

Runs inside the isolated lcm-infer container. Loads the released
tokenizer.pt + generator.pt, generates a few unconditional samples,
and measures teacher-forced per-level CE on real text (the decisive
comparator for our boundary-CE diagnostic on our own tokenizer).
"""
import os, sys, pickle, types, importlib
import torch

# --- stub hydra (only hydra.utils.get_class is used) ---
hydra = types.ModuleType("hydra")
class _HydraUtils:
    @staticmethod
    def get_class(path):
        mod, _, name = path.rpartition(".")
        return getattr(importlib.import_module(mod), name)
hydra.utils = _HydraUtils
sys.modules["hydra"] = hydra

ROOT = "/repo/vendor/ncm-upstream"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer

CK = "/work/ncm_official"
CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)

torch.set_num_threads(int(os.environ.get("NTHREADS", "10")))

tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model)
sd = torch.load(os.path.join(CK, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
print("keys:", [k for k in sd.keys() if k != "model"])
L0_P = sd["inference"]["level0_unigram"].float() if "inference" in sd else None
print("l0 unigram:", None if L0_P is None else L0_P.shape)

EOS = tk.pre_tokenizer.eos_token_id
PAD = tk.pre_tokenizer.pad_token_id
levels = net.levels
print("levels:", levels, "vocab_sizes:", net.vocab_sizes)

# ---------- 1) official samples (ground truth) ----------
torch.manual_seed(0)
B = 4
L0 = levels[0]
l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0) if L0_P is not None else None
prefix = torch.full((B, 1), EOS, dtype=torch.long)
g = net.generate(l0, prefix, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0)
_, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
for r in ids:
    print("OFFICIAL:", tk.pre_tokenizer.decode([t for t in r if t not in (EOS, PAD)])[:260])

# ---------- 2) teacher-forced per-level CE on real text ----------
# TinyStories test text: use their released tokenized data if present,
# else encode raw text lines through their BPE.
texts = [
    "Once upon a time there was a little girl named Lily. She loved to play in the garden with her dog. One day she found a shiny stone. She kept it in her pocket and showed it to her friends. They were all very happy and played together until sunset.",
    "Tom had a red ball. He took it to the park every day. His friend Sam had a blue kite. They played together near the big tree. Suddenly the wind blew the kite away. Tom ran fast and caught it. Sam was very grateful and shared his snacks.",
    "A little bird lived in a tall tree. Every morning it sang a sweet song. The cat under the tree listened quietly. One day the bird fell from its nest. The cat did not hurt it. Instead the cat helped the bird climb back. They became good friends forever.",
    "Mom made a big cake for the party. It had yellow icing and many candles. All the children clapped and smiled. The baby tried to blow the candles but the wind was too strong. Everyone laughed and helped him. The cake was sweet and soft inside.",
]
enc = [torch.tensor(tk.pre_tokenizer.encode(t)[:128]) for t in texts]
from torch.nn.utils.rnn import pad_sequence
ids = pad_sequence(enc, batch_first=True, padding_value=PAD)[:, :128]
ids = torch.nn.functional.pad(ids, (0, 128 - ids.shape[1]), value=PAD)
with torch.no_grad():
    chunks, targets = tk.vqvae.get_ncm_inputs(ids)  # chunks (B,640,384); targets = codes levels 1..K
    z_prefix = tk.vqvae.encoder.wte(ids[:, :1])     # raw_token_prefix: token emb at vq dim
    x = torch.cat([z_prefix, chunks], dim=1)        # [prefix | ncm inputs]
    logits = net(x)
print("targets:", targets.shape, "logits:", logits.shape)
offs = 0
for i, l in enumerate(levels):
    if i == 0 and not net.predict_level0:
        # targets include level0? check offset layout
        pass
# figure out target layout: get_ncm_inputs returned idx[:, levels[0]:]
tl = targets  # (B, L - l0) presumably
print("L-l0:", sum(levels[1:]), "targets dim:", tl.shape)
pos = 0
ces = {}
for i, l in enumerate(levels[1:]):
    seg = logits[:, pos:pos + l, :]
    tgt = tl[:, pos:pos + l]
    V = net.vocab_sizes[i] if hasattr(net, 'vocab_sizes') else seg.shape[-1]
    ce = torch.nn.functional.cross_entropy(seg.reshape(-1, seg.shape[-1]), tgt.reshape(-1)).item()
    ces[f"l{i+1}"] = (l, V, round(ce, 3))
    pos += l
import math
print("per-level teacher-forced CE (uniform = ln V):")
for k, (l, V, ce) in ces.items():
    print(f"  {k}: L={l:3d} V={V} ce={ce:.3f} unif={math.log(V):.2f}")
EOF_MARKER = None
