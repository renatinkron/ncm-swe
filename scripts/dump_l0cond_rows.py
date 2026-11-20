"""Dump segment->code-row alignment + prompt prefixes for the l0cond probe.

Reproduces encode_official.py's load_records + segs_of EXACTLY (same corpus
iteration order and sha1 8% split), so emitted row i lines up with
artifacts/codes_official/{train,held}.npy row i.

For each 128-token segment we emit the first K official-BPE token ids and
their decoded text — the "prompt" the l0cond adapter conditions on.

  podman run --rm -i --network none -v $PWD:/work -w /work \
    localhost/lcm-infer python3 scripts/dump_l0cond_rows.py
"""
import hashlib, json, os, sys, types, importlib

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
os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer

CK = "/work/artifacts/ncm_official"
CORPUS = os.environ.get("CORPUS", "/repo/data/corpus_ts_vq")
OUT = "/work/artifacts/l0cond"
CTX = 128
PREFIX_LEN = int(os.environ.get("PREFIX_LEN", "16"))


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


def dump_stream(recs, tk, path):
    """Exact mirror of segs_of: one continuous token stream across records,
    cut into CTX segments (a segment may span a record boundary). For each
    emitted segment we keep the first PREFIX_LEN token ids + decoded text.
    The record id stored is the one the segment STARTED in."""
    eos = tk.pre_tokenizer.eos_token_id
    pad = tk.pre_tokenizer.pad_token_id
    n = 0
    with open(path, "w") as f:
        cur = []
        cur_rec = None
        for r in recs:
            if not cur:
                cur_rec = r["id"]  # record where the pending segment starts
            cur += tk.pre_tokenizer.encode(r["text"]) + [eos]
            while len(cur) >= CTX:
                seg = cur[:CTX]
                cur = cur[CTX:]
                pre = seg[:PREFIX_LEN]
                rec = {
                    "row_idx": n,
                    "rec_id": cur_rec,
                    "prefix_ids": pre,
                    "prefix_text": tk.pre_tokenizer.decode(pre),
                    "n_pad": 0,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                cur_rec = r["id"]  # next segment starts inside this record
        if cur:
            seg = cur + [pad] * (CTX - len(cur))
            pre = seg[:PREFIX_LEN]
            rec = {
                "row_idx": n,
                "rec_id": cur_rec,
                "prefix_ids": pre,
                "prefix_text": tk.pre_tokenizer.decode(pre),
                "n_pad": CTX - len(cur),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(
        CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(
        CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(
        CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)

    os.makedirs(OUT, exist_ok=True)
    recs = load_records(CORPUS)
    tr, ho = split(recs)
    print(f"records: {len(recs)} train {len(tr)} held {len(ho)}", flush=True)

    for name, split_recs in [("train", tr), ("held", ho)]:
        path = os.path.join(OUT, f"rows_{name}.jsonl")
        n = dump_stream(split_recs, tk, path)
        print(f"{name}: {n} rows -> {path}", flush=True)

    # sanity: counts must match codes_official meta
    meta = json.load(open("/work/artifacts/codes_official/meta.json"))
    print("expected train/held:",
          meta["train_shape"][0], meta["held_shape"][0])


if __name__ == "__main__":
    main()
