"""Eval profunda do gen_bpe_func256: diversidade, filler breakdown,
argmax com prefixes variados, e top_p vs top_k."""
import ast, json, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/repo/scripts")
import _bpe_probe as P  # noqa: E402

OUTD = "/work/artifacts/ncm_code"


def gen(model, fast, seed_ids, n, temp, top_k, top_p, seed):
    torch.manual_seed(seed)
    model.eval()
    idx = torch.tensor([list(seed_ids)] * n, dtype=torch.long)
    with torch.no_grad():
        for _ in range(P.CTX - len(seed_ids)):
            logits = model(idx)[:, -1]
            if temp <= 0:
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


def breakdown(text):
    """filler: docstring vs comment vs other-string share of chars"""
    import io, tokenize
    doc = com = oth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.STRING and tok.start[1] in (0, 4):
                doc += len(tok.string)
            elif tok.type == tokenize.COMMENT:
                com += len(tok.string)
            elif tok.type == tokenize.STRING:
                oth += len(tok.string)
    except Exception:
        pass
    t = max(len(text), 1)
    return round(doc / t, 3), round(com / t, 3), round(oth / t, 3)


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    model = P.BPELM()
    ck = os.environ.get("CKPT_NAME", "gen_bpe_func256.pt")
    stem = ck.replace(".pt", "")
    model.load_state_dict(torch.load(OUTD + "/" + ck, map_location="cpu",
                                     weights_only=False)["model"])
    from tokenizers import Tokenizer as HFTokenizer
    fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")
    res = {}
    t0 = time.time()

    # 1) argmax com prefixes variados
    for px in ["def ", "class ", "def test_", "def _", "async def "]:
        sids = [0] + fast.encode(px).ids
        rows = gen(model, fast, sids, 8, 0.0, 0, 0.0, 7)
        rows = [r[1:] if r and r[0] == 0 else r for r in rows]
        p_, r_, ff = P.eval_gen(fast, rows)
        res[f"argmax[{px!r}]"] = {"parse": p_, "real": r_, "filler": ff}
        print(f"argmax {px!r}: parse {p_}/8 real {r_}/8 t={time.time()-t0:.0f}s",
              flush=True)
        if px == "def ":
            json.dump([P.decode(fast, r) for r in rows],
                      open(OUTD + f"/{stem}_deep_argmax_def.json", "w"), indent=1)

    # 2) sampled top_k24 vs top_p0.9 — seed [0]+'def '
    sids = [0] + fast.encode("def ").ids
    for tag, tk, tp in [("topk24", 24, 0.0), ("topp09", 0, 0.9)]:
        rows = gen(model, fast, sids, P.N_EVAL, 0.6, tk, tp, 7)
        rows = [r[1:] if r and r[0] == 0 else r for r in rows]
        texts = [P.decode(fast, r) for r in rows]
        p_, r_, ff = P.eval_gen(fast, rows)
        parsed = [t for t in texts if _parses(t)]
        uniq = len(set(parsed))
        bds = [breakdown(t) for t in parsed]
        res[tag] = {"parse": p_, "real": r_, "filler": ff,
                    "uniq_parsed": uniq,
                    "doc_mean": round(sum(b[0] for b in bds) / max(len(bds), 1), 3),
                    "com_mean": round(sum(b[1] for b in bds) / max(len(bds), 1), 3),
                    "str_mean": round(sum(b[2] for b in bds) / max(len(bds), 1), 3)}
        print(f"{tag}: parse {p_}/32 real {r_}/32 uniq {uniq} "
              f"t={time.time()-t0:.0f}s", flush=True)
        json.dump(texts, open(OUTD + f"/{stem}_deep_{tag}.json", "w"), indent=1)

    res["ckpt"] = ck
    json.dump(res, open(OUTD + f"/{stem}_deep_metrics.json", "w"), indent=1)
    print("done", json.dumps(res)[:600], flush=True)


def _parses(t):
    try:
        ast.parse(t)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
