"""REPL interativo p/ gen_bpe_*: digite um prefixo de código,
recebe 1 argmax + N variantes amostradas. Comandos:
  :n K        — nº de variantes sampled (default 4)
  :temp T     — temperatura (default 0.6)
  :topk K     — top_k (default 24)   :topp P — top_p (default 0.9, 0=off)
  :mode a|s|b — só argmax / só sampled / ambos (default b)
  :q          — sair
"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/repo/scripts")
import _bpe_probe as P  # noqa: E402

OUTD = "/work/artifacts/ncm_code"
CKPT = os.environ.get("CKPT_NAME", "gen_bpe_curr.pt")

torch.manual_seed(int(os.environ.get("SEED", "0")))
model = P.BPELM()
model.load_state_dict(torch.load(OUTD + "/" + CKPT,
                                 map_location="cpu", weights_only=False)["model"])
model.eval()
from tokenizers import Tokenizer as HFTokenizer  # noqa: E402
fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")
print(f"[{CKPT} · {sum(p.numel() for p in model.parameters())/1e6:.1f}M · "
      f"ctx {P.CTX}] pronto — prefixo ou :help", flush=True)


def gen(seed_ids, n, temp, top_k, top_p, greedy):
    idx = torch.tensor([list(seed_ids)] * n, dtype=torch.long)
    with torch.no_grad():
        for _ in range(P.CTX - len(seed_ids)):
            logits = model(idx)[:, -1]
            if greedy or temp <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                lg = logits / temp
                if top_k:
                    v, _ = lg.topk(top_k)
                    lg = lg.masked_fill(lg < v[:, -1:], -1e9)
                if top_p:
                    s, si = lg.sort(-1, descending=True)
                    cum = F.softmax(s, -1).cumsum(-1)
                    drop = cum - F.softmax(s, -1)[:, :1] > top_p
                    s[drop] = -1e9
                    lg = torch.zeros_like(lg).scatter(-1, si, s)
                nxt = torch.multinomial(F.softmax(lg, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if (nxt == 0).all():
                break
    return idx.tolist()


n_s, temp, top_k, top_p, mode = 4, 0.6, 24, 0.9, "b"
while True:
    try:
        line = input("\n> ").rstrip("\n")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not line:
        continue
    if line.startswith(":"):
        parts = line.split()
        c = parts[0]
        if c == ":q":
            break
        try:
            if c == ":n":
                n_s = int(parts[1])
            elif c == ":temp":
                temp = float(parts[1])
            elif c == ":topk":
                top_k = int(parts[1])
            elif c == ":topp":
                top_p = float(parts[1])
            elif c == ":mode":
                mode = parts[1]
            print(f"[n={n_s} temp={temp} topk={top_k} topp={top_p} mode={mode}]")
        except Exception:
            print("comandos: :n :temp :topk :topp :mode a|s|b :q")
        continue
    t0 = time.time()
    seed = [0] + fast.encode(line).ids
    if mode in ("a", "b"):
        rows = gen(seed, 1, 0.0, 0, 0.0, greedy=True)
        txt = P.decode(fast, [i for i in rows[0][1:] if i != 1])
        print(f"── argmax ({time.time()-t0:.1f}s)\n{txt}")
    if mode in ("s", "b"):
        t0 = time.time()
        rows = gen(seed, n_s, temp, top_k, top_p, greedy=False)
        for i, r in enumerate(rows):
            txt = P.decode(fast, [x for x in r[1:] if x != 1])
            print(f"── s{i} ({time.time()-t0:.1f}s)\n{txt}")
