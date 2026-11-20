"""Paired conditional-generation eval for the l0cond adapter (container only).

For each of N held prefixes: adapter predicts an l0 distribution from the
frozen-Qwen feature of the prefix; gen_ncm_v2 continues generation with
generate(l0=...). Paired arm: same prefixes, l0 ~ unigram (official recipe).
Metrics per arm: prompt_cosine (cosine of frozen-Qwen hidden-mean features,
prefix vs generated text — same instrument as the adapter), dup_rate,
distinct_ratio. Also logs adapter top-k hit vs the held row's true l0.

  podman run --rm -i --network none \
    -v $PWD:/repo -v $PWD/artifacts:/work/artifacts \
    -v $HOME/models:/models:ro -w /repo localhost/lcm-infer \
    python3 scripts/bench_l0cond.py \
      --adapter /work/artifacts/l0cond/adapter_H.pt --n 16 \
      --out /work/artifacts/bench_l0cond.json
"""
import argparse, json, math, os, sys, types, importlib

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
sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/scripts")
os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
import numpy as np
import torch
import torch.nn as nn

from ncm_swe.model.generator import LevelGenerator
from train_level_heads import FrozenQwen
from l0cond_features import iter_rows

CK = "/work/artifacts/ncm_official"
L0C = "/work/artifacts/l0cond"
MODEL_DIR = os.environ.get("QWEN_DIR", "/models/qwen05-sft")


def pick_tokens(logits):
    return logits.argmax(-1)


def rep_stats(texts):
    dup = tot = 0
    ratios = []
    for t in texts:
        w = t.split()
        dup += sum(1 for a, b in zip(w, w[1:]) if a == b)
        tot += max(len(w) - 1, 0)
        ratios.append(len(set(w)) / max(len(w), 1))
    return round(dup / max(tot, 1), 4), round(
        sum(ratios) / max(len(ratios), 1), 4)


def make_head(arch, d, v):
    if arch == "linear":
        return nn.Linear(d, v)
    return nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, v))


def feat_of(texts, tk_q, model_q, emb, kind, max_len=64):
    enc = tk_q(texts, return_tensors="pt", padding=True, truncation=True,
               max_length=max_len)
    ids, mask = enc.input_ids, enc.attention_mask
    with torch.no_grad():
        if kind == "E":
            f = emb[ids] * mask.unsqueeze(-1)
        else:
            f = model_q(ids) * mask.unsqueeze(-1)
    return f.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--gen", default="/work/artifacts/gen_ncm_v2/final.pt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--l0-topk", type=int, default=64,
                    help="top-k truncation on the adapter l0 distribution")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "10")))

    ad = torch.load(args.adapter, map_location="cpu", weights_only=False)
    head = make_head(ad["arch"], ad["d"], ad["v"])
    head.load_state_dict(ad["sd"])
    head.eval()
    kind = ad["feature"]
    print(f"adapter: arch={ad['arch']} feature={kind}", flush=True)

    from transformers import AutoTokenizer
    tk_q = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_q = FrozenQwen(MODEL_DIR).eval()
    emb = model_q.embed.weight

    rows = list(iter_rows(f"{L0C}/rows_held.jsonl"))[:args.n]
    prefixes = [r["prefix_text"] for r in rows]
    held = np.load("/work/artifacts/codes_official/held.npy",
                   mmap_mode="r")
    true_l0 = np.asarray(held[:args.n, 0], dtype=np.int64)

    feats = feat_of(prefixes, tk_q, model_q, emb, kind, max_len=24)
    with torch.no_grad():
        lg = head(feats)                                    # (n, 4096)
    # adapter diagnostics: rank of true l0 + top-k hit
    rank = (lg > lg.gather(1, torch.tensor(true_l0).unsqueeze(1))
            ).sum(1).tolist()
    t8 = (lg.topk(8, -1).indices == torch.tensor(true_l0).unsqueeze(1)
          ).any(-1).float().mean().item()
    # l0 distribution: top-k truncation then softmax
    lg_k = lg.clone()
    if args.l0_topk > 0:
        thr = lg_k.topk(args.l0_topk, -1).values[:, -1:]
        lg_k = lg_k.masked_fill(lg_k < thr, float("-inf"))
    l0_adapter = torch.multinomial(lg_k.softmax(-1), 1)    # (n,1) -> l0 pos
    print(f"adapter: median rank of true l0={sorted(rank)[len(rank)//2]} "
          f"top8_hit={t8:.3f}", flush=True)

    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(
        CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(
        CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(
        CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vqvae = tk.vqvae.eval()

    ck = torch.load(args.gen, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = LevelGenerator(vqvae, n_embd=cfg.get("n_embd", 896),
                           n_layer=cfg.get("n_layer", 4), n_head_q=14,
                           n_head_kv=2, mlp_hidden=4864,
                           intra=cfg.get("intra", False),
                           corrupt=cfg.get("corrupt", 0.0))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    L0 = model.levels[0]
    p_uni = np.load("/work/artifacts/l0_unigram.npy").astype(np.float64)
    p_uni = torch.tensor(p_uni / p_uni.sum()).float()
    l0_unigram = torch.multinomial(
        p_uni, args.n * L0, replacement=True).view(args.n, L0)

    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

    def gen_decode(l0):
        # reseed per arm so both arms share the same within-level sampling
        # stream — paired comparison must not leak the first arm's draws
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        idx = model.generate(batch_size=args.n, temperature=args.temp,
                             top_k=args.top_k, top_p=1.0, l0=l0)
        with torch.no_grad():
            _, zl = vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
            ids = pick_tokens(vqvae.decoder(zl[-1]))
        texts = []
        for row in ids.tolist():
            if EOS in row:
                row = row[:row.index(EOS)]
            texts.append(tk.pre_tokenizer.decode(
                [t for t in row if t not in (EOS, PAD)],
                skip_special_tokens=True))
        return idx, texts

    arms = {}
    for name, l0 in [("adapter", l0_adapter), ("unigram", l0_unigram)]:
        t0 = __import__("time").time()
        idx, texts = gen_decode(l0)
        dup, dist = rep_stats(texts)
        fg = feat_of(texts, tk_q, model_q, emb, kind, max_len=64)
        cos = torch.nn.functional.cosine_similarity(feats, fg, dim=-1)
        arms[name] = {"l0_codes": idx[:, :L0].tolist(), "texts": texts,
                      "n_empty": sum(1 for t in texts if not t.strip()),
                      "prompt_cosine": [round(float(c), 4) for c in cos],
                      "mean_cosine": round(float(cos.mean()), 4),
                      "dup_rate": dup, "distinct_ratio": dist}
        print(f"{name}: cos={arms[name]['mean_cosine']:.3f} "
              f"dup={dup:.3f} dist={dist:.3f}", flush=True)

    res = {"n": args.n, "temp": args.temp, "topk": args.top_k,
           "seed": args.seed, "adapter": os.path.basename(args.adapter),
           "feature": kind, "l0_topk": args.l0_topk,
           "adapter_true_l0_median_rank":
               sorted(rank)[len(rank) // 2],
           "adapter_true_l0_top8": round(t8, 4),
           "prefixes": prefixes, "arms": arms}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
