"""Bench the official-codes NCM generator vs recorded AR baseline (container).

Generates n samples with the trained LevelGenerator on official codes,
decodes via vendored vqvae + official BPE, saves samples + timing JSON.
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
import torch
from ncm_swe.model.generator import LevelGenerator

CK = "/work/artifacts/ncm_official"
GEN = os.environ.get("GEN", "/work/artifacts/gen_ncm_official/final.pt")
OUT = os.environ.get("BENCH_OUT", "/work/artifacts/bench_official.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("THREADS", "10")))

    CFG = OmegaConf.load(os.path.join(ROOT, "config/eval/tinystories.yaml"))
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    CFG.tokenizer.pre_tokenizer.vocab_file_path = os.path.join(CK, CFG.checkpoints.bpe_vocab)
    CFG.tokenizer.pre_tokenizer.merges_file_path = os.path.join(CK, CFG.checkpoints.bpe_merges)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    vqvae = tk.vqvae.eval()

    assert os.path.isfile(GEN), f"GEN must be an existing checkpoint file, got {GEN!r}"
    ck = torch.load(GEN, map_location="cpu", weights_only=False)
    c = ck["config"]
    model = LevelGenerator(vqvae, n_embd=c["n_embd"], n_layer=c["n_layer"],
                           corrupt=0.0)
    model.load_state_dict(ck["model"])
    model.eval()

    t0 = time.time()
    idx = model.generate(batch_size=args.n, temperature=args.temp,
                         top_k=args.topk)
    gen_s = time.time() - t0
    texts = model.decode_text(idx, tk.pre_tokenizer)
    res = {"n": args.n, "temp": args.temp, "topk": args.topk,
           "seed": args.seed, "gen_s": round(gen_s, 2),
           "passes": int(model.last_passes),
           "s_per_sample": round(gen_s / args.n, 3),
           "texts": texts}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "texts"}))
    for t in texts:
        print("---", t[:220].replace("\n", " "))

main()
