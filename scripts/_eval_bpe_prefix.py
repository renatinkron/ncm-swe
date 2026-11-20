"""Eval prefix-seeded do gen_bpe_func.pt: gera a partir de 'def ' (como
completion) vs a partir de token 0 (OOD). Separa 'modelo nao gera' de
'start conditioning errado'."""
import json, os, sys, time
import torch
sys.path.insert(0, "/repo/scripts")
import _bpe_probe as P  # noqa: E402

OUTD = "/work/artifacts/ncm_code"


def gen_prefixed(model, fast, prefix, n, temp, top_k, seed):
    ids0 = ([0] if os.environ.get("BOS") == "1" else []) + \
        fast.encode(prefix).ids
    torch.manual_seed(seed)
    model.eval()
    idx = torch.tensor([ids0] * n, dtype=torch.long)
    import torch.nn.functional as F
    with torch.no_grad():
        for _ in range(P.CTX - len(ids0)):
            logits = model(idx)[:, -1]
            if temp <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
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
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    model = P.BPELM()
    ck_name = os.environ.get("CKPT_NAME", "gen_bpe_func.pt")
    stem = ck_name.replace(".pt", "")
    model.load_state_dict(torch.load(
        OUTD + "/" + ck_name, map_location="cpu",
        weights_only=False)["model"])
    from tokenizers import Tokenizer as HFTokenizer
    fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")
    res = {}
    t0 = time.time()
    prefix = os.environ.get("PREFIX", "def ")
    for tag, temp, topk in [("argmax", 0.0, 0), ("sampled", 0.6, 24)]:
        rows = gen_prefixed(model, fast, prefix, P.N_EVAL, temp, topk, 7)
        rows = [r[1:] if r and r[0] == 0 else r for r in rows]
        texts = [P.decode(fast, r) for r in rows]
        p_, r_, ff = P.eval_gen(fast, rows)
        res["prefix_" + tag] = {"parse": p_, "real": r_,
                                "filler_frac": ff}
        print(tag, res["prefix_" + tag], f"t={time.time()-t0:.0f}s",
              flush=True)
        json.dump(texts, open(OUTD + f"/{stem}_gen_{tag}.json", "w"),
                  indent=1)
    res["ckpt"] = ck_name
    res["prefix"] = prefix
    res["bos"] = os.environ.get("BOS") == "1"
    json.dump(res, open(OUTD + f"/{stem}_prefix_metrics.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
