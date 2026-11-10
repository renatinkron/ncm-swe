"""Train LevelGenerator on OFFICIAL NCM codes (container only).

Wraps vendored tk.vqvae directly — our LevelGenerator duck-types it
(vq.embeddings/_up_to_full/_apply_quant_resi/idx_to_ncm_input + decode).
corrupt=1.0 => per-item rate ~ U(0,1), mean 0.5 (official recipe).
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
import torch.nn.functional as F
from ncm_swe.model.generator import LevelGenerator

CK = "/work/artifacts/ncm_official"
CODES = "/work/artifacts/codes_official"
OUT = os.environ.get("OUT_DIR", "/work/artifacts/gen_ncm_official")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corrupt", type=float, default=1.0)
    ap.add_argument("--corrupt-schedule", default="flat",
                    choices=["flat", "linear"],
                    help="flat: r_i~U(0,corrupt) por item | linear: taxa por nível corrupt-hi→corrupt-lo (oficial)")
    ap.add_argument("--corrupt-hi", type=float, default=0.75)
    ap.add_argument("--corrupt-lo", type=float, default=0.25)
    ap.add_argument("--init-from", default="",
                    help="checkpoint p/ warm-start (resume)")
    ap.add_argument("--offset", type=int, default=0,
                    help="offset do step p/ metrics contínuas")
    ap.add_argument("--log-every", type=int, default=10)
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
    for p in vqvae.parameters():
        p.requires_grad_(False)

    tr = np.load(f"{CODES}/train.npy").astype(np.int64)
    ho = np.load(f"{CODES}/held.npy").astype(np.int64)
    print(f"train {tr.shape} held {ho.shape} levels {vqvae.vq.levels}", flush=True)

    model = LevelGenerator(vqvae, n_embd=896, n_layer=4, n_head_q=14,
                           n_head_kv=2, mlp_hidden=4864,
                           corrupt=args.corrupt)
    model.corrupt_schedule = args.corrupt_schedule
    model.corrupt_hi = args.corrupt_hi
    model.corrupt_lo = args.corrupt_lo
    if args.init_from:
        prev = torch.load(args.init_from, map_location="cpu",
                          weights_only=False)
        model.load_state_dict(prev["model"], strict=True)
        print(f"warm-start de {args.init_from}", flush=True)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"generator trainable params {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    os.makedirs(OUT, exist_ok=True)
    mlog = open(f"{OUT}/metrics.jsonl", "a")
    t0 = time.time()
    for step in range(args.steps):
        bi = np.random.randint(0, len(tr), args.batch)
        idx = torch.tensor(tr[bi], dtype=torch.long)
        loss, per = model.loss(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            rec = {"step": step + args.offset,
                 "loss": round(float(loss), 4),
                   **per, "s": round(time.time() - t0)}
            mlog.write(json.dumps(rec) + "\n"); mlog.flush()
            print(rec, flush=True)
        if step % 500 == 499:
            torch.save({"model": model.state_dict(),
                        "config": {"n_embd": 896, "n_layer": 4,
                                   "corrupt": args.corrupt,
                                   "corrupt_schedule": args.corrupt_schedule,
                                   "corrupt_hi": args.corrupt_hi,
                                   "corrupt_lo": args.corrupt_lo,
                                   "levels": list(vqvae.vq.levels)},
                        "model_type": "official_codes"},
                       f"{OUT}/ckpt.pt")
    # held eval — dois protocolos: como-treinado (corrupt on) e limpo
    with torch.no_grad():
        bi = np.random.randint(0, len(ho), min(64, len(ho)))
        batch = torch.tensor(ho[bi], dtype=torch.long)
        hl, hper = model.loss(batch)
        model.corrupt = 0.0
        hc, hcper = model.loss(batch)
    torch.save({"model": model.state_dict(),
                "config": {"n_embd": 896, "n_layer": 4,
                           "corrupt": args.corrupt,
                           "corrupt_schedule": args.corrupt_schedule,
                           "corrupt_hi": args.corrupt_hi,
                           "corrupt_lo": args.corrupt_lo,
                           "levels": list(vqvae.vq.levels)},
                "model_type": "official_codes"}, f"{OUT}/final.pt")
    mlog.write(json.dumps({"step": args.steps + args.offset,
                           "held_loss": round(float(hl), 4),
                           **{f"held_{k}": v for k, v in hper.items()},
                           "held_clean_loss": round(float(hc), 4),
                           **{f"held_clean_{k}": v for k, v in hcper.items()},
                           "wall_s": round(time.time() - t0, 1)}) + "\n")
    mlog.close()
    print(f"done wall={time.time()-t0:.0f}s held_loss={float(hl):.4f} {hper}")

main()
