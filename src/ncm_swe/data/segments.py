"""Corpus → pre-token id stream → fixed-length segments.

Split is at RECORD level (sha1 of record id), before tokenization, so the
held-out slice can never leak into training shards.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_records(corpus_dir: str | Path) -> list[dict]:
    recs = []
    for shard in sorted(Path(corpus_dir).glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def split_records(recs: list[dict], heldout_pct: int = 8) -> tuple[list[dict], list[dict]]:
    train, held = [], []
    for r in recs:
        bucket = int(hashlib.sha1(r["id"].encode()).hexdigest()[:4], 16) % 100
        (held if bucket < heldout_pct else train).append(r)
    return train, held


def tokenize_records(recs: list[dict], bpe, eos_sep: bool = True) -> np.ndarray:
    """Concatenate pre-token ids of all records into one uint32 stream."""
    parts = []
    for r in recs:
        ids = bpe.encode(r["text"])
        if eos_sep:
            ids = ids + [bpe.eos_token_id]
        parts.append(np.asarray(ids, dtype=np.uint32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)


def segments(stream: np.ndarray, context_length: int,
             pad_id: int = 0) -> np.ndarray:
    """Slice token stream into (N, context_length) int64 segments (pad tail)."""
    n_full = len(stream) // context_length
    out = stream[: n_full * context_length].reshape(n_full, context_length)
    if len(stream) % context_length:
        tail = np.full(context_length, pad_id, dtype=np.uint32)
        tail[: len(stream) % context_length] = stream[n_full * context_length:]
        out = np.vstack([out, tail[None, :]])
    return out.astype(np.int64)


def build_segments(corpus_dir, bpe, context_length: int,
                   heldout_pct: int = 8) -> tuple[np.ndarray, np.ndarray, dict]:
    recs = load_records(corpus_dir)
    train, held = split_records(recs, heldout_pct)
    meta = {"records": len(recs), "train_records": len(train),
            "held_records": len(held)}
    tr = segments(tokenize_records(train, bpe), context_length, bpe.pad_token_id)
    ho = segments(tokenize_records(held, bpe), context_length, bpe.pad_token_id)
    meta["train_segments"] = len(tr)
    meta["held_segments"] = len(ho)
    meta["train_tokens"] = int(tr.size)
    return tr, ho, meta
