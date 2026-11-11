"""Encode corpus_ts_vq records into OFFICIAL NCM VQ-VAE codes.

Runs ONLY inside the lcm-infer container (vendored third-party code).
Output: artifacts/codes_official/{train,held}.npy (N,641 uint16) + meta.json.
Same record split (sha1 id, 8% held) as the local pipeline -> same partition.
"""
import hashlib, json, os, sys, time, types, importlib
import numpy as np
import torch

hydra = types.ModuleType("hydra")
class _U:
    @staticmethod
    def get_class(p):
        m, _, n = p.rpartition(".")
        return getattr(importlib.import_module(m), n)
hydra.utils = _U
sys.modules["hydra"] = hydra

ROOT = "/repo/vendor/ncm-upstream"
REPO = "/repo"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer

CK = "/work/artifacts/ncm_official"
CORPUS = os.environ.get("CORPUS", "/work/corpus_ts_vq")
OUT = "/work/artifacts/codes_official"
CTX = 128
BATCH = 128

def load_records(d):
    recs = []
    lim = int(os.environ.get("SHARDS", "0")) or 10**9
    for shard in sorted(__import__("pathlib").Path(d).glob("shard-*.jsonl"))[:lim]:
        for line in shard.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs

def split(recs, pct=8):
    tr, ho = [], []
    for r in recs:
        b = int(hashlib.sha1(r["id"].encode()).hexdigest()[:4], 16) % 100
        (ho if b < pct else tr).append(r)
    return tr, ho

def segs_of(recs, tk):
    eos = tk.pre_tokenizer.eos_token_id
    pad = tk.pre_tokenizer.pad_token_id
    out = []
    cur = []
    for r in recs:
        cur += tk.pre_tokenizer.encode(r["text"]) + [eos]
        while len(cur) >= CTX:
            out.append(cur[:CTX]); cur = cur[CTX:]
    if cur:
        out.append(cur + [pad] * (CTX - len(cur)))
    return out

def encode_segments(vq, segs):
    codes = np.zeros((len(segs), 641), dtype=np.uint16)
    for i in range(0, len(segs), BATCH):
        ids = torch.tensor(segs[i:i+BATCH], dtype=torch.long)
        with torch.no_grad():
            idx = vq.encode(ids)
        codes[i:i+len(idx)] = idx.cpu().numpy().astype(np.uint16)
        if (i // BATCH) % 100 == 0:
            print(f"  enc {i}/{len(segs)}", flush=True)
    return codes

def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "10")))
    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vq = tk.vqvae.eval()

    t0 = time.time()
    recs = load_records(CORPUS)
    tr, ho = split(recs)
    print(f"records {len(recs)} -> train {len(tr)} held {len(ho)}", flush=True)
    tr_s, ho_s = segs_of(tr, tk), segs_of(ho, tk)
    print(f"segments train {len(tr_s)} held {len(ho_s)} tok_t={time.time()-t0:.0f}s", flush=True)
    tr_c, ho_c = encode_segments(vq, tr_s), encode_segments(vq, ho_s)
    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/train.npy", tr_c)
    np.save(f"{OUT}/held.npy", ho_c)
    lvls = vq.vq.levels
    uniq = []
    ofs = 0
    for l in lvls:
        uniq.append(int(np.unique(tr_c[:, ofs:ofs+l]).size))
        ofs += l
    json.dump({"levels": list(lvls), "vocab": [4096]*len(lvls),
               "context_length": CTX, "records": len(recs),
               "train_records": len(tr), "held_records": len(ho),
               "train_shape": list(tr_c.shape), "held_shape": list(ho_c.shape),
               "uniq_per_level": uniq, "wall_s": round(time.time()-t0, 1)},
              open(f"{OUT}/meta.json", "w"), indent=1)
    print("uniq_per_level:", uniq, "wall", round(time.time()-t0), "s")

main()
