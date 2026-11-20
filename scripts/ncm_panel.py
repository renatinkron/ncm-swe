"""NCM render-panel: controllable generation over the official NCM checkpoint.

Maps renderer-style controls onto the released TinyStories NCM model
(artifacts/ncm_official): level-0 source, temperature/top-k/top-p, classifier-free
guidance, repetition penalties (freq/mu/adj, ported from eval_shard.pick_tokens),
per-level preview decode, and text-quality metrics. CPU-only, container-safe.

  selftest : python3 scripts/ncm_panel.py --selftest        (no ckpt needed)
  smoke    : podman run --rm --network none -v $PWD:/repo -w /repo \
               -e THREADS=8 localhost/lcm-infer:latest \
               python3 scripts/ncm_panel.py --n 4 --out /work/artifacts/panel_smoke.json
"""
import argparse, json, math, os, sys, time, types, importlib

import torch
import torch.nn.functional as F

ROOT = "/repo/vendor/ncm-upstream"
if not os.path.isdir(ROOT):
    ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "vendor", "ncm-upstream")
CK_DEFAULT = "/work/artifacts/ncm_official"
if not os.path.isdir(CK_DEFAULT):
    CK_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "artifacts", "ncm_official")


# ---------------- pure helpers (unit-tested, no ckpt) ----------------

def dup4_rate(texts):
    """Mean fraction of duplicated word 4-grams per text (0 = no repetition)."""
    rates = []
    for t in texts:
        w = t.split()
        grams = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
        if not grams:
            continue
        rates.append(1.0 - len(set(grams)) / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def unicode_frac(text):
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def distinct_ratio(text):
    w = text.split()
    return len(set(w)) / len(w) if w else 0.0


def text_metrics(texts):
    return {
        "dup4_rate": round(dup4_rate(texts), 4),
        "unicode_frac": round(sum(unicode_frac(t) for t in texts) / max(1, len(texts)), 4),
        "distinct_ratio": round(sum(distinct_ratio(t) for t in texts) / max(1, len(texts)), 4),
        "mean_chars": round(sum(len(t) for t in texts) / max(1, len(texts)), 1),
    }


def pick_tokens(logits, freq=0.0, mu=0.0, adj=None, allow=None):
    """Greedy per-position token pick with repetition penalties.

    Port of vendor/ncm-upstream/scripts/eval_shard.py::pick_tokens.
    logits: (B, T, V) decoder logits -> (B, T) token ids.
    freq: penalty per excess occurrence; mu: boost under-used tokens (needs allow);
    adj: (alpha, window, decay) penalizes recent picks; allow: per-token allowance.
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
                              torch.full((B, 1), -al * ad ** j, dtype=torch.float64,
                                         device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    return out


def l0_source(mode, l0_prob, batch, n_l0, vocab, generator=None):
    """Level-0 code source. Returns (B, n_l0) long tensor or None (= model head)."""
    if mode == "model":
        return None
    if mode == "unigram":
        p = l0_prob
    elif mode == "uniform":
        p = torch.ones(vocab, dtype=torch.float64) / vocab
    elif mode == "active":
        p = (l0_prob.double() > 0).double()
        p = p / p.sum()
    elif mode.startswith("smooth:"):
        eps = float(mode.split(":", 1)[1])
        p = (1.0 - eps) * l0_prob.double() + eps * torch.ones(vocab, dtype=torch.float64) / vocab
    else:
        raise ValueError(f"--l0 inválido: {mode}")
    return torch.multinomial(p.float(), batch * n_l0, replacement=True).view(batch, n_l0)


def parse_adj(s):
    if not s:
        return None
    return [float(v) for v in s.split(",")]


def selftest():
    g = torch.Generator().manual_seed(0)
    texts = ["the cat sat on the mat and the cat sat on the mat again and again",
             "um dois três quatro cinco seis sete oito nove dez onze doze treze",
             "a b c d e f g h i j k l m n o p q r s t u v w x y z"]
    m = text_metrics(texts)
    assert 0.0 <= m["dup4_rate"] <= 1.0 and m["mean_chars"] > 0
    assert text_metrics(["", "   "])["distinct_ratio"] == 0.0
    logits = torch.randn(2, 5, 11)
    ids = pick_tokens(logits)
    assert ids.shape == (2, 5)
    ids2 = pick_tokens(logits, freq=1.0, adj=(0.5, 2, 0.9))
    assert ids2.shape == (2, 5)
    p = torch.tensor([0.5, 0.5] + [0.0] * 9)
    l0 = l0_source("smooth:0.1", p, 2, 3, 11, generator=g)
    assert l0.shape == (2, 3) and l0.min() >= 0 and l0.max() < 11
    l0u = l0_source("uniform", p, 2, 3, 11, generator=g)
    assert l0u.shape == (2, 3)
    assert parse_adj("0.5,2,0.9") == [0.5, 2.0, 0.9]
    print("selftest OK")


# ---------------- ckpt-backed generation ----------------

def load_panel(ckpt_dir, threads):
    torch.set_num_threads(threads)
    hydra = types.ModuleType("hydra")
    class _U:
        @staticmethod
        def get_class(p):
            m, _, n = p.rpartition(".")
            return getattr(importlib.import_module(m), n)
    hydra.utils = _U
    sys.modules.setdefault("hydra", hydra)
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from omegaconf import OmegaConf
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    from models.next_concept import NextConceptTransformer

    cfg = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ckpt_dir, cfg.checkpoints.tokenizer)
    cfg.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(ckpt_dir, cfg.checkpoints.bpe_vocab)
    cfg.tokenizer.pre_tokenizer.merges_file_path = os.path.join(ckpt_dir, cfg.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.eval()
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model)
    sd = torch.load(os.path.join(ckpt_dir, cfg.checkpoints.generator),
                    map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    net.eval()
    l0_p = sd["inference"]["level0_unigram"].float() if "inference" in sd else None
    return tk, net, l0_p


def main():
    ap = argparse.ArgumentParser(description="NCM render-panel over official ckpt")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default=os.environ.get("NCM_CKPT", CK_DEFAULT))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "8")))
    ap.add_argument("--l0", default="unigram",
                    help="unigram|uniform|active|smooth:<eps>|model")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--cfg", type=float, default=0.0, help="classifier-free guidance scale")
    ap.add_argument("--freq", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--adj", default="", help="alpha,window,decay e.g. 0.5,2,0.9")
    ap.add_argument("--preview", action="store_true",
                    help="also emit per-level preview decodes")
    ap.add_argument("--kv", action="store_true", help="use KV cache in generate")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.out:
        args.out = os.path.abspath(args.out)  # load_panel chdirs into vendor/
    tk, net, l0_p = load_panel(args.ckpt, args.threads)
    torch.manual_seed(args.seed)
    levels = net.levels
    B = args.n
    eos = tk.pre_tokenizer.eos_token_id
    pad = tk.pre_tokenizer.pad_token_id

    if args.l0 == "model":
        if not (getattr(net, "predict_level0", False) or getattr(net, "query_level0", False)):
            raise SystemExit("--l0 model needs a ckpt with predict_level0 or query_level0; "
                             "this one has neither — use --l0 unigram|uniform|active|smooth:<eps>")
        l0 = None
    else:
        if l0_p is None:
            raise SystemExit("ckpt has no level0_unigram; --l0 model requires predict_level0/query_level0")
        l0 = l0_source(args.l0, l0_p, B, levels[0], l0_p.numel()).long()
    prefix = torch.full((B, 1), eos, dtype=torch.long)

    t0 = time.time()
    with torch.no_grad():
        idx = net.generate(l0, prefix, temperature=args.temp, top_k=args.top_k,
                           top_p=args.top_p, use_kv_cache=args.kv,
                           cfg_scale=args.cfg, cfg_null_eos=eos)
        _, zl = tk.vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
        logits = tk.vqvae.decoder(zl[-1])
        freq, mu, adj = args.freq, args.mu, parse_adj(args.adj)
        allow = None
        if mu > 0:
            allow = torch.ones(logits.shape[-1])
        ids = pick_tokens(logits, freq=freq, mu=mu, adj=adj, allow=allow)
    gen_s = time.time() - t0

    eos_t, pad_t = int(eos), int(pad)
    texts = []
    for r in ids.tolist():
        if eos_t in r:
            r = r[:r.index(eos_t)]
        texts.append(tk.pre_tokenizer.decode([t for t in r if t not in (eos_t, pad_t)],
                                             skip_special_tokens=True))

    previews = None
    if args.preview:
        rec = tk.decode_multiscales(idx)
        previews = ["\n".join(f"[L{i}] {p}" for i, p in enumerate(levels_txt))
                    for levels_txt in rec]

    res = {
        "ckpt": args.ckpt, "n": B, "seed": args.seed,
        "l0": args.l0, "temp": args.temp, "top_k": args.top_k, "top_p": args.top_p,
        "cfg": args.cfg, "freq": args.freq, "mu": args.mu, "adj": args.adj,
        "gen_s": round(gen_s, 2), "s_per_sample": round(gen_s / B, 3),
        "passes": len(levels), "levels": list(levels),
        "metrics": text_metrics(texts), "texts": texts,
    }
    if previews is not None:
        res["previews"] = previews
    blob = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        open(args.out, "w").write(blob)
    print(json.dumps({k: v for k, v in res.items() if k not in ("texts", "previews")},
                     ensure_ascii=False))
    for t in texts:
        print("---", t[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
