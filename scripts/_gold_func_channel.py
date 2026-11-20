"""Channel sanity: encode func units gold -> decode -> parse rate.
Separa 'canal quebrado' de 'previsao quebrada'."""
import ast, json, os, sys
import torch
sys.path.insert(0, "/repo/scripts")
from ncm_codetrain import build_tk, decode_codes  # noqa: E402

OUTD = "/work/artifacts/ncm_code"


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    tune = torch.load(OUTD + "/vq_code_tune.pt", weights_only=False)
    tk, vq = build_tk(tune)
    import pickle
    from tokenization.bpe_tokenizer import BPETokenizer
    EOS = chr(60) + "|endoftext|>"
    PAD = chr(60) + "|pad|>"
    bpe = BPETokenizer(pickle.load(open(OUTD + "/bpe_code_vocab.pickle", "rb")),
                       pickle.load(open(OUTD + "/bpe_code_merges.pickle", "rb")),
                       {"eos": EOS, "pad": PAD})
    import numpy as np
    codes = np.load(OUTD + "/codes_func.npy")
    units = [u for u in open(OUTD + "/func_units.txt").read().split("\n\n")
             if u.strip()]
    n = 32
    gold_codes = codes[-n:]
    gold_texts = units[len(codes) - n:len(codes)]
    dec_texts = [decode_codes(vq, bpe, r.astype("int64"))
                 for r in gold_codes]
    n_parse_gold = sum(1 for t in gold_texts if _ok(t))
    n_parse_dec = sum(1 for t in dec_texts if _ok(t))
    # fidelidade texto
    import difflib
    sims = [difflib.SequenceMatcher(None, g, d).ratio()
            for g, d in zip(gold_texts, dec_texts)]
    res = {"n": n, "gold_parse": n_parse_gold, "dec_parse": n_parse_dec,
           "text_sim_mean": round(sum(sims) / len(sims), 3)}
    print(json.dumps(res), flush=True)
    json.dump({"res": res, "gold": gold_texts[:3], "dec": dec_texts[:3]},
              open(OUTD + "/gold_func_channel.json", "w"), indent=1)


def _ok(t):
    try:
        ast.parse(t)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
