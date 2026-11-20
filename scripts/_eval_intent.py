"""Avalia o canal intenção→código: seed = comentário de intenção,
gera sampled (topk24 t0.6) e mede parse/real + relevância lexical
(a função gerada tem nome/corpo relacionado à intenção?)."""
import ast, json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/repo/scripts")
import _bpe_probe as P  # noqa: E402
from tokenizers import Tokenizer as HFTokenizer  # noqa: E402

OUTD = "/work/artifacts/ncm_code"
CKPT = os.environ.get("CKPT_NAME", "gen_bpe_currv3.pt")
N_GEN = int(os.environ.get("N_GEN", "8"))

INTENTS = [
    "# calcula o pi",
    "# ordena a lista",
    "# fibonacci",
    "# conta a frequencia de cada item",
    "# soma os valores",
    "# le o arquivo",
    "# busca binaria",
    "# compute the moving average",
    "# merge two dicts",
    "# palindrome check",
    "# estrutura de pilha",
    "# asynchronously fetch items",
]
# relevância lexical: palavras que a implementação correta tenderia a ter
KEYWORDS = {
    "# calcula o pi": ["pi", "4", "range", "sum"],
    "# ordena a lista": ["sort", "sorted"],
    "# fibonacci": ["fib", "a, b", "b, a"],
    "# conta a frequencia de cada item": ["count", "freq", "get("],
    "# soma os valores": ["sum", "acc", "total", "+="],
    "# le o arquivo": ["open", "read"],
    "# busca binaria": ["mid", "lo", "hi", "// 2", "binary"],
    "# compute the moving average": ["avg", "sum(", "k)", "window"],
    "# merge two dicts": ["{**", "update", "merge", "dict("],
    "# palindrome check": ["[::-1]", "palin"],
    "# estrutura de pilha": ["class", "push", "pop", "append"],
    "# asynchronously fetch items": ["async", "await"],
}


def gen(model, seed_ids, n, temp=0.6, top_k=24):
    torch.manual_seed(7)
    model.eval()
    idx = torch.tensor([list(seed_ids)] * n, dtype=torch.long)
    with torch.no_grad():
        for _ in range(P.CTX - len(seed_ids)):
            logits = model(idx)[:, -1]
            lg = logits / temp
            if top_k:
                v, _ = lg.topk(top_k)
                lg = lg.masked_fill(lg < v[:, -1:], -1e9)
            nxt = torch.multinomial(F.softmax(lg, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if (nxt == 0).all():
                break
    return idx.tolist()


def main():
    ck = os.environ.get("CKPT_NAME", CKPT)
    model = P.BPELM()
    model.load_state_dict(torch.load(OUTD + "/" + ck, map_location="cpu",
                                     weights_only=False)["model"])
    fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")
    stem = ck.rsplit(".", 1)[0]
    res, all_txt = {}, {}
    for intent in INTENTS:
        t0 = time.time()
        sids = [0] + fast.encode(intent + "\n").ids
        rows = gen(model, sids, N_GEN)
        texts = [P.decode(fast, [i for i in r[1:] if i != 1]) for r in rows]
        parse = sum(1 for t in texts if _ok(t))
        kws = KEYWORDS[intent]
        rel = sum(1 for t in texts
                  if _ok(t) and any(k.lower() in t.lower() for k in kws))
        res[intent] = {"parse": parse, "real": parse, "relevant": rel,
                       "n": N_GEN}
        all_txt[intent] = texts
        print(f"{intent!r}: parse {parse}/{N_GEN} relevant {rel} "
              f"t={time.time()-t0:.0f}s", flush=True)
    stem_ck = os.environ.get("CKPT_NAME", "currv3").rsplit(".", 1)[0]
    json.dump(res, open(OUTD + f"/{stem_ck}_intent_metrics.json", "w"),
              indent=1)
    json.dump(all_txt, open(OUTD + f"/{stem_ck}_intent_samples.json", "w"),
              indent=1)
    print("done", flush=True)


def _ok(t):
    try:
        tree = ast.parse(t)
    except Exception:
        return False
    return any(not (isinstance(n, ast.Expr) and
                    isinstance(n.value, ast.Constant)) for n in tree.body)


if __name__ == "__main__":
    main()
