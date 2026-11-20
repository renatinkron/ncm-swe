"""Extract frozen-Qwen features for the l0cond probe (host: ~/.venvs/ncm).

For each row of artifacts/l0cond/rows_{train,held}.jsonl we tokenize
prefix_text with the Qwen tokenizer and produce two feature arms:

  E (embed-mean): mean of embed_tokens rows over the prefix — lookup only,
     the cheapest feature that carries pretrained semantics. All rows.
  H (hidden-mean): mean-pool of the FrozenQwen last hidden state over the
     prefix — the full frozen-backbone representation. --max-h rows only.

Labels: l0 code = codes_official/{split}.npy[:, 0] (1 position, 4096 classes).

  ~/.venvs/ncm/bin/python scripts/l0cond_features.py [--max-h 40000]
"""
import argparse, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_level_heads import FrozenQwen  # noqa: E402

MODEL_DIR = os.environ.get("QWEN_DIR", os.path.expanduser("~/models/qwen05-sft"))
ROWS = "artifacts/l0cond"
CODES = "artifacts/codes_official"
OUT = "artifacts/l0cond"


def iter_rows(path):
    """jsonl reader robust to U+2028/U+2029 inside strings (json.dumps does
    not escape them, so naive per-line parsing breaks). raw_decode walks the
    stream skipping whitespace, so a record may span 'lines'."""
    dec = json.JSONDecoder()
    buf = open(path, encoding="utf-8").read()
    i, n = 0, len(buf)
    while i < n:
        while i < n and buf[i].isspace():
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(buf, i)
        yield obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-h", type=int, default=40000,
                    help="rows for the H (full forward) arm, train split")
    ap.add_argument("--max-h-held", type=int, default=0,
                    help="rows for the H arm, held split (0 = max-h)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("THREADS", "10")))

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = FrozenQwen(MODEL_DIR).eval()
    emb = model.embed.weight  # (V, 896)

    for split in ["train", "held"]:
        rows = list(iter_rows(f"{ROWS}/rows_{split}.jsonl"))
        codes = np.load(f"{CODES}/{split}.npy", mmap_mode="r")
        assert len(rows) == codes.shape[0], (len(rows), codes.shape)
        l0 = np.asarray(codes[:, 0], dtype=np.int64)
        np.save(f"{OUT}/l0_{split}.npy", l0)

        texts = [r["prefix_text"] for r in rows]
        t0 = time.time()

        # ---- arm E: embed-mean over ALL rows (lookup only) ----
        feat_e = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
        hmax = args.max_h if split == "train" else (
            args.max_h_held or args.max_h)
        hcap = min(hmax, len(texts))
        feat_h = np.zeros((hcap, emb.shape[1]), dtype=np.float32)
        for i in range(0, len(texts), args.batch):
            chunk = texts[i:i + args.batch]
            enc = tk(chunk, return_tensors="pt", padding=True,
                     truncation=True, max_length=24)
            ids, mask = enc.input_ids, enc.attention_mask
            # E arm: masked mean of embeddings
            e = emb[ids] * mask.unsqueeze(-1)
            feat_e[i:i + ids.shape[0]] = (
                e.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)).numpy()
            # H arm: forward on the subsample only
            if i < hcap:
                with torch.no_grad():
                    h = model(ids)                       # B×L×H
                h = h * mask.unsqueeze(-1)
                hi = min(i + ids.shape[0], hcap)
                feat_h[i:hi] = (
                    h[:hi - i].sum(1)
                    / mask[:hi - i].sum(1, keepdim=True).clamp_min(1)).numpy()
            if (i // args.batch) % 200 == 0:
                print(f"[{split}] {i}/{len(texts)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        np.save(f"{OUT}/feat_E_{split}.npy", feat_e)
        np.save(f"{OUT}/feat_H_{split}.npy", feat_h)
        print(f"[{split}] done E{feat_e.shape} H{feat_h.shape} "
              f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
