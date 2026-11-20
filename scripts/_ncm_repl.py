"""REPL interativo do NCM oficial (TinyStories) — a visão 'renderizador'.

Carrega o ckpt uma vez; cada 'g' gera n amostras com os knobs atuais.
Comandos:
  :n 4        :temp 0.9    :topk 50     :topp 0.9
  :cfg 2.0    :freq 0.3    :mu 0.5      :adj 0.5,2,0.9
  :l0 unigram|uniform|active|smooth:0.05|model
  :seed 42    :preview on|off    :kv on|off    :q
Enter ou 'g' = gerar.
"""
import os
import sys
import time

import torch

sys.path.insert(0, "/repo/scripts")
import ncm_panel as NP  # noqa: E402


def gen(tk, net, l0_p, S):
    torch.manual_seed(S["seed"])
    levels = net.levels
    B = S["n"]
    eos = tk.pre_tokenizer.eos_token_id
    pad = tk.pre_tokenizer.pad_token_id
    if S["l0"] == "model":
        l0 = None
    else:
        l0 = NP.l0_source(S["l0"], l0_p, B, levels[0], l0_p.numel()).long()
    prefix = torch.full((B, 1), eos, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        idx = net.generate(l0, prefix, temperature=S["temp"],
                           top_k=S["top_k"], top_p=S["top_p"],
                           use_kv_cache=S["kv"],
                           cfg_scale=S["cfg"], cfg_null_eos=eos)
        _, zl = tk.vqvae.vq.idx_to_zhats(idx, return_zlevels=True)
        logits = tk.vqvae.decoder(zl[-1])
        allow = torch.ones(logits.shape[-1]) if S["mu"] > 0 else None
        ids = NP.pick_tokens(logits, freq=S["freq"], mu=S["mu"],
                             adj=NP.parse_adj(S["adj"]), allow=allow)
    dt = time.time() - t0
    if S["preview"]:
        rec = tk.decode_multiscales(idx)
        for i, lv in enumerate(rec):
            print(f"── amostra {i} níveis ──")
            for j, p in enumerate(lv):
                print(f"  [L{j}] {p}")
    eos_t, pad_t = int(eos), int(pad)
    for i, r in enumerate(ids.tolist()):
        if eos_t in r:
            r = r[:r.index(eos_t)]
        txt = tk.pre_tokenizer.decode(
            [t for t in r if t not in (eos_t, pad_t)],
            skip_special_tokens=True)
        print(f"── final {i} ──\n{txt}")
    print(f"[{dt:.2f}s · {dt/B:.3f}s/amostra · {len(levels)} passes]")


def main():
    ck = os.environ.get("NCM_CKPT", "/work/artifacts/ncm_official")
    threads = int(os.environ.get("THREADS", "4"))
    print(f"carregando NCM ({ck})…", flush=True)
    tk, net, l0_p = NP.load_panel(ck, threads)
    print(f"pronto · níveis {net.levels} · digite 'g' ou Enter\n")
    S = {"n": 4, "seed": 0, "temp": 1.0, "top_k": 50, "top_p": 0.9,
         "cfg": 0.0, "freq": 0.0, "mu": 0.0, "adj": "",
         "l0": "unigram", "preview": False, "kv": False}
    while True:
        try:
            line = input(
                f"n{S['n']} t{S['temp']} k{S['top_k']} p{S['top_p']} "
                f"cfg{S['cfg']} l0={S['l0']}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("", "g"):
            try:
                gen(tk, net, l0_p, S)
            except Exception as e:
                print("erro:", e)
            continue
        if line == ":q":
            break
        if line.startswith(":seed "):
            S["seed"] = int(line.split()[1]); continue
        if line.startswith(":preview "):
            S["preview"] = line.split()[1] == "on"; continue
        if line.startswith(":kv "):
            S["kv"] = line.split()[1] == "on"; continue
        if line.startswith(":adj "):
            S["adj"] = line.split()[1]; continue
        if line.startswith(":l0 "):
            S["l0"] = line.split()[1]; continue
        for k, cast in (("n", int), ("temp", float), ("topk", int),
                        ("topp", float), ("cfg", float), ("freq", float),
                        ("mu", float)):
            if line.startswith(f":{k} "):
                key = {"topk": "top_k", "topp": "top_p"}.get(k, k)
                S[key] = cast(line.split()[1])
                break
        else:
            print("comandos: :n :temp :topk :topp :cfg :freq :mu :adj "
                  ":l0 :seed :preview :kv :q — 'g' gera")


if __name__ == "__main__":
    main()
