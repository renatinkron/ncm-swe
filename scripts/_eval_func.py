"""Eval endurecida do gen_func1.pt: argmax (top_k=1) e sampled (temp0.9/topk32).
Oráculo: ast.parse + >=1 stmt real (nao-docstring/constante) + filler_frac.
"""
import ast, io, json, os, pickle, sys, tokenize
import torch

sys.path.insert(0, "/repo/src")
sys.path.insert(0, "/repo/scripts")
from ncm_codetrain import build_tk, decode_codes  # noqa: E402
from tokenization.bpe_tokenizer import BPETokenizer  # noqa: E402
from ncm_swe.model.generator import LevelGenerator  # noqa: E402

OUTD = "/work/artifacts/ncm_code"


def n_real(text):
    try:
        tree = ast.parse(text)
    except Exception:
        return -1
    return sum(1 for nd in tree.body
               if not (isinstance(nd, ast.Expr)
                       and isinstance(nd.value, ast.Constant)))


def filler_fraction(text):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
        fill = sum(len(t.string) for t in toks
                   if t.type in (tokenize.COMMENT, tokenize.STRING))
        return fill / max(len(text), 1)
    except Exception:
        return 0.0


def score(texts):
    n_parse = n_real_ok = 0
    fills = []
    for t in texts:
        try:
            ast.parse(t)
            n_parse += 1
            if n_real(t) >= 1:
                n_real_ok += 1
        except Exception:
            pass
        fills.append(filler_fraction(t))
    return {"parse": n_parse, "real": n_real_ok,
            "filler_frac": round(sum(fills) / len(fills), 3)}


def gen_rows(gen, n, seed, temperature, top_k):
    torch.manual_seed(seed)
    gen.eval()
    idx = gen.generate(batch_size=n, temperature=temperature,
                       top_k=top_k, top_p=1.0)
    return [r.tolist() for r in idx]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    tune = torch.load(OUTD + "/vq_code_tune.pt", weights_only=False)
    tk, vq = build_tk(tune)
    EOS = chr(60) + "|endoftext|>"
    PAD = chr(60) + "|pad|>"
    spec = {"eos": EOS, "pad": PAD}
    bpe = BPETokenizer(pickle.load(open(OUTD + "/bpe_code_vocab.pickle", "rb")),
                       pickle.load(open(OUTD + "/bpe_code_merges.pickle", "rb")),
                       spec)

    init = os.environ.get("INIT", OUTD + "/gen_code3.pt")
    ck0 = torch.load(init, map_location="cpu", weights_only=False)
    gcfg = ck0.get("config", {})
    gen = LevelGenerator(vq, n_embd=gcfg.get("n_embd", 896),
                         n_layer=gcfg.get("n_layer", 4), n_head_q=14,
                         n_head_kv=2, mlp_hidden=4864,
                         intra=gcfg.get("intra", False),
                         corrupt=gcfg.get("corrupt", 0.0))
    ck = os.environ.get("GEN_CK", OUTD + "/gen_func1.pt")
    gen.load_state_dict(torch.load(ck, map_location="cpu",
                                   weights_only=False), strict=True)

    res = {"ck": os.path.basename(ck)}
    for name, temp, k in [("argmax", 1.0, 1), ("sampled", 0.9, 32)]:
        rows = gen_rows(gen, 32, 7, temp, k)
        texts = [decode_codes(vq, bpe, r) for r in rows]
        res[name] = score(texts)
        json.dump(texts, open(OUTD + f"/eval_func_{name}.json", "w"),
                  indent=1)
        print(name, res[name], flush=True)
    json.dump(res, open(OUTD + "/eval_func_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
