#!/usr/bin/env python
"""Oracle Conditioning Probe — alimenta l0 REAL (do encoder) e mede se
os niveis gerados l1..l5 produzem codigo coerente.

Se parsear -> o gargalo insoluvel e a geracao nao-AR do l0.
Se sopa    -> o acoplamento residual l1..l5 tambem esta quebrado.
"""
import ast
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ncm_swe.tokenizer.bpe import CorpusBPE
from ncm_swe.vq.model import load_vqvae
from bench_generate import load_gen

vqvae, _, _ = load_vqvae("artifacts/tokenizer_probe.pt")
vqvae.eval()
bpe = CorpusBPE.load("artifacts/bpe.json")
gen, _ = load_gen("artifacts/gen_ncm_probe/final.pt", vqvae)

codes = np.load("artifacts/codes_probe/train.npy")
L0 = gen.levels[0]
print(f"codes {codes.shape} | L0 cols = {L0}")

n = 32
idx = np.random.RandomState(0).choice(len(codes), n, replace=False)
l0_real = torch.from_numpy(codes[idx][:, :L0]).long()
print(f"l0_real {l0_real.shape} range [{l0_real.min()},{l0_real.max()}]")

torch.manual_seed(0)
t0 = time.time()
with torch.no_grad():
    out, _ = gen.generate(batch_size=n, temperature=0.0, top_k=1,
                          return_per_level=True, l0=l0_real)
print(f"[generate] {time.time()-t0:.2f}s")

texts = gen.decode_text(out, bpe)
ok = 0
for i, txt in enumerate(texts):
    try:
        ast.parse(txt)
        ok += 1
    except SyntaxError:
        pass
    if i < 3:
        print(f"--- amostra {i} ---\n{txt[:280]}")
print(f"\nORACLE RESULT: {ok}/{n} parseable com l0 ground-truth")
