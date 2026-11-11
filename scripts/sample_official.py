"""Sample from our LevelGenerator trained on OFFICIAL NCM codes (container only).

Mirrors eval_shard.py's generation/decode path, plus the l0-source modes:
  --l0 model    -> l0 predicted by the model head (our pre-existing behavior)
  --l0 unigram  -> l0 ~ empirical corpus marginal (official default)
  --l0 uniform|active|smooth:<eps> -> upstream ablations

  podman run --rm -i --network none -v $PWD:/work -w /work localhost/lcm-infer \
    python3 scripts/sample_official.py --gen artifacts/gen_ncm_v2/final.pt \
    --l0 unigram --n 8 --out artifacts/bench_l0uni.json
"""
import argparse, json, os, sys, time, types, importlib

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
os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
import numpy as np
import torch
from ncm_swe.model.generator import LevelGenerator

CK = "/work/artifacts/ncm_official"


def l0_dist(mode, path, vocab):
    if mode == "model":
        return None
    p = np.load(path, allow_pickle=False).astype(np.float64)
    p = torch.tensor(p / p.sum())
    if mode == "unigram":
        pass
    elif mode == "uniform":
        p = torch.ones_like(p) / p.numel()
    elif mode == "active":
        p = (p > 0).double(); p = p / p.sum()
    elif mode.startswith("smooth:"):
        eps = float(mode.split(":", 1)[1])
        p = (1.0 - eps) * p + eps * torch.ones_like(p) / p.numel()
    else:
        raise SystemExit(f"--l0 inválido: {mode}")
    assert p.numel() == vocab, f"unigram {p.numel()} != vocab {vocab}"
    return p.float()


def pick_tokens(logits, freq=0.0, mu=0.0, adj=None, allow=None):
    """Port of upstream eval_shard.pick_tokens — decode-time anti-repetition.

    freq: penalizes already-picked tokens ∝ count (or count-allowance).
    mu:   boosts under-used tokens early (needs allow).
    adj:  (al, aw, ad) — last `aw` picks penalized al*ad^j each step.
    allow: per-token allowance vector (scaled by T) or None → excess=count.
    """
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


def rep_stats(texts):
    """Adjacent-duplicate word rate + distinct-word ratio (repetition proxies)."""
    dup = tot = 0
    ratios = []
    for t in texts:
        w = t.split()
        dup += sum(1 for a, b in zip(w, w[1:]) if a == b)
        tot += max(len(w) - 1, 0)
        ratios.append(len(set(w)) / max(len(w), 1))
    return round(dup / max(tot, 1), 4), round(sum(ratios) / max(len(ratios), 1), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l0", default="model")
    ap.add_argument("--l0-unigram", default="/work/artifacts/l0_unigram.npy")
    ap.add_argument("--freq", type=float, default=0.0,
                    help="frequency_penalty no decode (excess=count; upstream)")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="frequency_deficit — boost em tokens sub-usados")
    ap.add_argument("--adj", default="",
                    help="adjacency_penalty 'al,aw,ad' — penaliza picks recentes")
    ap.add_argument("--intra-passes", type=int, default=0,
                    help="override do chunk size no decode intra (0=cfg)")
    ap.add_argument("--refine-iters", type=int, default=0,
                    help="rodadas de mask-predict refinement por nível")
    ap.add_argument("--refine-frac", type=float, default=0.3,
                    help="fração menos confiante remascarada por rodada")
    ap.add_argument("--refine-from", type=int, default=1,
                    help="nível mínimo (índice) onde refinement se aplica")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "10")))

    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vqvae = tk.vqvae.eval()
    held = np.load("/work/artifacts/codes_official/held.npy").astype(np.int64)

    ck = torch.load(args.gen, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = LevelGenerator(vqvae, n_embd=cfg.get("n_embd", 896),
                           n_layer=cfg.get("n_layer", 4), n_head_q=14,
                           n_head_kv=2, mlp_hidden=4864,
                           intra=cfg.get("intra", False),
                           intra_passes=args.intra_passes
                           or cfg.get("intra_passes", 32),
                           intra_mask=cfg.get("intra_mask", "boundary"),
                           corrupt=cfg.get("corrupt", 0.0))
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    L0 = model.levels[0]
    p = l0_dist(args.l0, args.l0_unigram, int(model.vocab_sizes[0]))
    l0_codes = torch.multinomial(p, args.n * L0, replacement=True).view(args.n, L0) \
        if p is not None else None

    t0 = time.time()
    idx = model.generate(batch_size=args.n, temperature=args.temp,
                         top_k=args.top_k, top_p=args.top_p, l0=l0_codes,
                         refine_iters=args.refine_iters,
                         refine_frac=args.refine_frac,
                         refine_from=args.refine_from)
    gen_s = time.time() - t0

    # decode: official path — decoder logits -> pick_tokens -> BPE decode (cut@EOS)
    adj = [float(v) for v in args.adj.split(",")] if args.adj else None
    with torch.no_grad():
        _, zl = vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
        ids = pick_tokens(vqvae.decoder(zl[-1]),
                          freq=args.freq, mu=args.mu, adj=adj)
        # exposure-bias probe: CE per level teacher-forced on SELF-generated
        # codes vs the same CE on ground-truth codes (diagnóstico direto).
        # corruption desligada nos dois lados — mede o modelo, não o ruído.
        model.corrupt = 0.0
        self_loss, self_per = model.loss(idx)
        bi = np.random.randint(0, len(held), min(len(idx), len(held)))
        gt_loss, gt_per = model.loss(torch.tensor(held[bi], dtype=torch.long))
    EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    texts = []
    for row in ids.tolist():
        if EOS in row:
            row = row[:row.index(EOS)]
        texts.append(tk.pre_tokenizer.decode(
            [t for t in row if t not in (EOS, PAD)], skip_special_tokens=True))

    dup_rate, dist_ratio = rep_stats(texts)
    res = {"n": args.n, "temp": args.temp, "topk": args.top_k,
           "top_p": args.top_p, "seed": args.seed, "l0_mode": args.l0,
           "freq": args.freq, "mu": args.mu, "adj": args.adj or None,
           "gen_s": round(gen_s, 2),
           "passes": getattr(model, "last_passes", len(model.levels)),
           "s_per_sample": round(gen_s / args.n, 3),
           "dup_rate": dup_rate, "distinct_ratio": dist_ratio,
           "ce_self_gen": {**self_per, "total": round(float(self_loss), 4)},
           "ce_held_gt": {**gt_per, "total": round(float(gt_loss), 4)},
           "l0_codes": idx[:, :L0].tolist(),
           "texts": texts}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "texts"}, indent=1))
    for t in texts:
        print("---", t[:300])


main()
