#!/usr/bin/env python3
"""Train a code-domain BPE (vocab 8192) and export in official NCM format.

Official format (artifacts/ncm_official):
  vocab.pickle  -> dict {int_id: bytes_token}; ids 0,1 = endoftext/pad,
                   ids 2..257 = single bytes, then merged tokens.
  merges.pickle -> list[(bytes, bytes)] in merge-rank order.

Trains with HF `tokenizers` ByteLevel BPE on data/corpus_code shards,
converts tokens back to raw bytes, measures chars/token vs old BPE.
"""
import glob
import json
import os
import pickle
import sys
import time

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "ncm_code")
CORPUS = os.path.join(ROOT, "data", "corpus_code")
OLD = os.path.join(ROOT, "artifacts", "ncm_official")
VOCAB = 8192
N_SHARDS = int(os.environ.get("SHARDS", "32"))

EOS = chr(60) + "|endoftext|>"
PAD = chr(60) + "|pad|>"


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def main():
    os.makedirs(OUT, exist_ok=True)
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    t2b = lambda s: bytes(u2b[c] for c in s)

    files = sorted(glob.glob(os.path.join(CORPUS, "shard_*.txt")))[:N_SHARDS]
    print(f"training BPE on {len(files)} shards", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(
        vocab_size=VOCAB, min_frequency=2,
        special_tokens=[EOS, PAD],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True)
    t0 = time.time()
    tok.train(files, tr)
    print(f"train time {time.time()-t0:.0f}s", flush=True)

    state = json.loads(tok.to_str())
    vocab_str = state["model"]["vocab"]           # {str: id}
    merges_str = state["model"]["merges"]         # ["a b", ...] rank order

    vocab = {i: t2b(s) for s, i in vocab_str.items()}
    # ensure specials hold ids 0,1 and every single byte exists
    assert vocab[0] == EOS.encode(), vocab[0]
    assert vocab[1] == PAD.encode(), vocab[1]
    merges = []
    for m in merges_str:
        a, b = m if isinstance(m, (list, tuple)) else m.partition(" ")[::2]
        merges.append((t2b(a), t2b(b)))

    tok.save(os.path.join(OUT, "bpe_code_tokenizer.json"))
    with open(os.path.join(OUT, "bpe_code_vocab.pickle"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(OUT, "bpe_code_merges.pickle"), "wb") as f:
        pickle.dump(merges, f)

    # measure chars/token on held-out code vs old BPE via official class
    sys.path.insert(0, os.path.join(ROOT, "vendor", "ncm-upstream"))
    from tokenization.bpe_tokenizer import BPETokenizer
    old_vocab = pickle.load(open(os.path.join(OLD, "bpe_vocab_8192.pickle"), "rb"))
    old_merges = pickle.load(open(os.path.join(OLD, "bpe_merges_8192.pickle"), "rb"))
    spec = {"eos": EOS, "pad": PAD}
    t_old = BPETokenizer(old_vocab, old_merges, spec)
    t_new = BPETokenizer(vocab, merges, spec)

    held = open(files[-1]).read() + open(sorted(glob.glob(
        os.path.join(CORPUS, "shard_*.txt")))[-1]).read()
    sample = held[:200_000]
    n_old = len(t_old.encode(sample))
    n_new = len(t_new.encode(sample))
    res = {
        "vocab_size": len(vocab), "n_merges": len(merges),
        "chars": len(sample), "train_s": round(time.time() - t0),
        "old_chars_per_tok": round(len(sample) / n_old, 2),
        "new_chars_per_tok": round(len(sample) / n_new, 2),
        "old_tokens": n_old, "new_tokens": n_new,
    }
    json.dump(res, open(os.path.join(OUT, "bpe_metrics.json"), "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
