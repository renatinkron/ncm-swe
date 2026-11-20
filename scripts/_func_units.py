"""Extrai unidades fatiadas por fronteira de função/classe do corpus de código.

Cada unidade começa num `def`/`class`/`async def` top-level (coluna 0) e vai
até o fim natural do bloco (próxima linha top-level não-vazia). Resultado:
unidades standalone parseáveis — o que o treino atual (slices mid-file)
não produz.

Saída:
  func_units.txt        unidades separadas por \n\n (formato LESSONS)
  func_units_stats.json contagens do funil
"""
import ast, glob, json, os, re

CORPUS = "/repo/data/corpus_code"
OUTD = "/work/artifacts/ncm_code"
START = re.compile(r"^(async def|def|class)\s")
MAX_TOK = int(os.environ.get("MAX_TOK", "127"))
OUT_NAME = os.environ.get("OUT_NAME", "func_units.txt")


def iter_blocks(text):
    """yield blocos top-level def/class por heurística de indentação"""
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        if START.match(lines[i]):
            start = i
            i += 1
            while i < n:
                ln = lines[i]
                if ln.strip() and ln[0] not in " \t#":
                    break
                i += 1
            yield "\n".join(lines[start:i])
        else:
            i += 1


def n_real(text):
    try:
        tree = ast.parse(text)
    except Exception:
        return -1
    return sum(1 for nd in tree.body
               if not (isinstance(nd, ast.Expr)
                       and isinstance(nd.value, ast.Constant)))


def squeeze_blank(unit):
    """remove linhas em branco internas (\n\n é separador do formato LESSONS)"""
    return "\n".join(l for l in unit.split("\n") if l.strip())


def main():
    from tokenizers import Tokenizer as HFTokenizer
    fast = HFTokenizer.from_file(OUTD + "/bpe_code_tokenizer.json")

    stats = {"blocks": 0, "parse_ok": 0, "real_ok": 0, "fit127": 0,
             "kept": 0, "tok_lens": []}
    units = []
    for f in sorted(glob.glob(CORPUS + "/shard_*.txt")):
        text = open(f, encoding="utf-8", errors="replace").read()
        for blk in iter_blocks(text):
            stats["blocks"] += 1
            u = squeeze_blank(blk)
            try:
                ast.parse(u)
            except Exception:
                continue
            stats["parse_ok"] += 1
            if n_real(u) < 1:
                continue
            stats["real_ok"] += 1
            n_tok = len(fast.encode(u).ids)
            if n_tok > MAX_TOK:
                continue
            stats["fit127"] += 1
            stats["tok_lens"].append(n_tok)
            units.append(u)

    # dedup exato e cap
    seen, out = set(), []
    for u in units:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    stats["kept"] = len(out)
    lens = stats.pop("tok_lens")
    stats["tok_mean"] = round(sum(lens) / max(len(lens), 1), 1)
    stats["tok_max"] = max(lens) if lens else 0
    stats["tok_min"] = min(lens) if lens else 0

    with open(OUTD + "/" + OUT_NAME, "w") as fo:
        fo.write("\n\n".join(out))
    stats["max_tok"] = MAX_TOK
    json.dump(stats, open(OUTD + "/" + OUT_NAME.replace(".txt", "_stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
