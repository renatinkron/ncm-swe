"""Plano B probe: transformer treinado DIRETO em tokens BPE (sem VQ-641).

Mesma pipeline de dados do ncm_codetrain (shards -> BPE ids -> rows 128),
next-token prediction. Comparação honesta: mesmo corpus, mesmo oráculo
endurecido (parse + stmt real), held-out rows 19000:20000.

Env:
  TRAIN_SECS  budget de parede do treino (default 10800 = 3h)
  THREADS     torch threads (default 4)
  N_TRAIN     rows de treino (default 19000)
  BATCH       batch size (default 16)
  LR          learning rate (default 3e-4 — LM direto aguenta mais que 1e-4)
  STEP_LOG    log a cada N steps (default 50)
  SMOKE=1     modo smoke: ~30 steps + eval mini, sem budget
"""
import ast, glob, json, os, pickle, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTD = "/work/artifacts/ncm_code"
CORPUS = "/repo/data/corpus_code"
CTX = int(os.environ.get("CTX", "128"))
VOCAB = 8192
N_TRAIN = int(os.environ.get("N_TRAIN", "19000"))
HOLDOUT = tuple(int(x) for x in
                os.environ.get("HOLDOUT", "19000,20000").split(","))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "3e-4"))
STEP_LOG = int(os.environ.get("STEP_LOG", "50"))
TRAIN_SECS = int(os.environ.get("TRAIN_SECS", "10800"))
N_EVAL = 32
CKPT = OUTD + "/" + os.environ.get("CKPT_NAME", "gen_bpe_probe.pt")


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class BPELM(nn.Module):
    def __init__(self, d=768, nl=6, nh=12):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        m = torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1)
        self.register_buffer("mask", m)

    def forward(self, idx):
        b, t = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(t, device=idx.device))
        for bl in self.blocks:
            x = bl(x, self.mask[:t, :t])
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, n, temp=0.0, top_k=0, seed=7):
        torch.manual_seed(seed)
        self.eval()
        idx = torch.zeros(n, 1, dtype=torch.long)
        for _ in range(CTX - 1):
            logits = self(idx)[:, -1]
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
        return idx[:, 1:].tolist()


def load_rows(n_rows):
    from tokenizers import Tokenizer as HFTokenizer
    fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")
    units_file = os.environ.get("UNITS_FILE")
    if units_file:
        stem = os.path.basename(units_file).replace(".txt", "")
        cache = OUTD + f"/bpe_{stem}_ctx{CTX}_rows.npy"
        if os.path.exists(cache):
            rows = np.load(cache)
            if len(rows) >= n_rows:
                return torch.tensor(rows[:n_rows].astype(np.int64)), fast
        units = [u for u in open(units_file).read().split("\n\n")
                 if u.strip()]
        rows_l = []
        for u in units:
            ids = [0] + fast.encode(u).ids[:CTX - 2] + [0]
            ids += [1] * (CTX - len(ids))
            rows_l.append(ids)
        rows = np.array(rows_l, dtype=np.uint16)
        np.save(cache, rows)
        return torch.tensor(rows.astype(np.int64)), fast
    cache = OUTD + f"/bpe_rows_ctx{CTX}.npy"
    if os.path.exists(cache):
        rows = np.load(cache)
        if len(rows) >= n_rows:
            return torch.tensor(rows[:n_rows].astype(np.int64)), fast
    ids_all = []
    for f in sorted(glob.glob(CORPUS + "/shard_*.txt")):
        ids_all.extend(fast.encode(open(f).read()).ids)
        if len(ids_all) > n_rows * CTX:
            break
    n = min(len(ids_all) // CTX, n_rows)
    rows = np.array([ids_all[i * CTX:(i + 1) * CTX]
                     for i in range(n)], dtype=np.uint16)
    np.save(cache, rows)
    return torch.tensor(rows.astype(np.int64)), fast


def decode(fast, ids):
    """id_to_token + byte-level unmap — inverso completo do bytes_to_unicode:
    char cp<256 = byte literal; cp>=256 = byte cp-256; redecoda utf-8."""
    if 0 in ids:
        ids = ids[:ids.index(0)]
    s = "".join(fast.id_to_token(i) or "" for i in ids)
    bs = bytes(c - 256 if c >= 256 else c for c in map(ord, s))
    return bs.decode("utf-8", errors="replace")


def n_real(text):
    try:
        tree = ast.parse(text)
    except Exception:
        return -1
    return sum(1 for n in tree.body
               if not (isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Constant)))


def eval_gen(fast, rows_ids):
    np_ = nr = 0
    filler = []
    for ids in rows_ids:
        t = decode(fast, ids)
        filler.append(filler_fraction(t))
        try:
            ast.parse(t)
            np_ += 1
            if n_real(t) >= 1:
                nr += 1
        except Exception:
            pass
    return np_, nr, round(sum(filler) / len(filler), 3) if filler else 0.0


def filler_fraction(text):
    """fraction of chars inside comments/strings (tokenize-based)"""
    import io, tokenize
    try:
        code = 0
        total = len(text)
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                code += len(tok.string)
        return code / total if total else 0.0
    except Exception:
        return 0.0


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    torch.manual_seed(int(os.environ.get("SEED", "7")))
    assert N_TRAIN <= HOLDOUT[0], "train rows must not overlap held-out"
    t0 = time.time()
    rows, fast = load_rows(HOLDOUT[1])
    train_rows = rows[:N_TRAIN]
    held = rows[HOLDOUT[0]:HOLDOUT[1]]
    print(f"rows train {tuple(train_rows.shape)} held {tuple(held.shape)}",
          flush=True)

    model = BPELM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e6:.2f}M", flush=True)
    init = os.environ.get("INIT")
    if init:
        model.load_state_dict(
            torch.load(init, map_location="cpu",
                       weights_only=False)["model"])
        print(f"warm-start {init}", flush=True)
    eval_only = os.environ.get("EVAL_ONLY") == "1"
    if eval_only:
        model.load_state_dict(
            torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    n = len(train_rows)
    step = 0
    losses = []
    smoke = os.environ.get("SMOKE") == "1"
    budget = 120 if smoke else TRAIN_SECS
    while not eval_only and time.time() - t0 < budget:
        bidx = torch.randint(0, n, (BATCH,))
        x = train_rows[bidx]
        logits = model(x)
        loss = ce(logits[:, :-1].reshape(-1, VOCAB),
                  x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        step += 1
        if step % STEP_LOG == 0:
            print(f"step {step} loss={np.mean(losses):.4f} "
                  f"t={int(time.time()-t0)}s", flush=True)
            losses = []
            torch.save({"model": model.state_dict()}, CKPT + ".tmp")
            os.replace(CKPT + ".tmp", CKPT)
        if smoke and step >= 30:
            break

    # held-out teacher-forced: CE + top-1 acc (métrica que o VQ não tem)
    model.eval()
    ce_sum = acc_sum = tokens = 0.0
    with torch.no_grad():
        for i in range(0, len(held), 64):
            lg = model(held[i:i + 64])
            tgt = held[i:i + 64, 1:]
            m = tgt.numel()
            ce_sum += ce(lg[:, :-1].reshape(-1, VOCAB),
                         tgt.reshape(-1)).item() * m
            acc_sum += (lg[:, :-1].argmax(-1) == tgt).float().sum().item()
            tokens += m
    hce = ce_sum / tokens
    acc = acc_sum / tokens

    res = {"steps": step, "train_secs": int(time.time() - t0),
           "lr": LR, "batch": BATCH, "n_train": N_TRAIN,
           "heldout": list(HOLDOUT), "units_file": os.environ.get("UNITS_FILE"),
           "heldout_rows": len(held),
           "heldout_ce": round(hce, 4), "heldout_top1": round(acc, 4)}
    for tag, temp, topk in [("argmax", 0.0, 0), ("sampled", 0.6, 24)]:
        ids = model.generate(N_EVAL, temp=temp, top_k=topk)
        p, r, ff = eval_gen(fast, ids)
        res[tag] = {"parse": p, "real": r, "filler_frac": ff}
        print(f"{tag}: parse {p}/{N_EVAL} real {r}/{N_EVAL} "
              f"filler {ff}", flush=True)

    out_name = os.environ.get(
        "METRICS_NAME",
        "bpe_probe_metrics_eval.json" if eval_only
        else "bpe_probe_metrics.json")
    json.dump(res, open(OUTD + "/" + out_name, "w"), indent=1)
    print("done", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
