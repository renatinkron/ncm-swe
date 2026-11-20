"""Validate the LoRA knowledge dump (data/lora_dump/knowledge_dump.jsonl).

  python3 scripts/dump_knowledge.py --check [path]

Schema per line: {"id": str, "lang": "en"|"pt", "domain": str, "topic": str,
"content": str}. Programming domains are English-only; logic/argumentation/
writing must exist in both languages.
"""
import argparse, json, sys
from collections import Counter

DEFAULT = "data/lora_dump/knowledge_dump.jsonl"
EN_ONLY_DOMAINS = {"algorithms", "data_structures", "patterns", "testing",
                   "debugging", "complexity"}
BILINGUAL_DOMAINS = {"logic", "argumentation", "writing"}
MIN_ENTRIES = 80
REQUIRED_FIELDS = {"id", "lang", "domain", "topic", "content"}


def validate(path):
    errors, rows = [], []
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        return [f"cannot open {path}: {e}"], []
    with fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: invalid JSON: {e}")
                continue
            missing = REQUIRED_FIELDS - set(r)
            if missing:
                errors.append(f"line {n}: missing fields {sorted(missing)}")
                continue
            if r["lang"] not in ("en", "pt"):
                errors.append(f"line {n}: bad lang {r['lang']!r}")
            if not all(isinstance(r[k], str) and r[k].strip() for k in REQUIRED_FIELDS):
                errors.append(f"line {n}: empty/non-string field")
            rows.append(r)
    ids = Counter(r["id"] for r in rows)
    errors += [f"duplicate id {i}" for i, c in ids.items() if c > 1]
    contents = Counter(r["content"] for r in rows)
    errors += [f"duplicate content at id {r['id']}" for r in rows
               if contents[r["content"]] > 1]
    langs = Counter(r["lang"] for r in rows)
    domains = Counter((r["domain"], r["lang"]) for r in rows)
    for d in EN_ONLY_DOMAINS:
        if domains[(d, "en")] == 0:
            errors.append(f"domain {d}: no 'en' entries")
        if domains[(d, "pt")] > 0:
            errors.append(f"domain {d}: unexpected 'pt' entries (programming is en-only)")
    for d in BILINGUAL_DOMAINS:
        for lg in ("en", "pt"):
            if domains[(d, lg)] == 0:
                errors.append(f"domain {d}: missing '{lg}' entries")
    if len(rows) < MIN_ENTRIES:
        errors.append(f"only {len(rows)} entries (< {MIN_ENTRIES})")
    stats = {"total": len(rows), "by_lang": dict(langs),
             "by_domain": {f"{d}/{l}": c for (d, l), c in sorted(domains.items())}}
    return errors, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("path", nargs="?", default=DEFAULT)
    args = ap.parse_args()
    if not args.check:
        ap.error("only --check is implemented")
    errors, stats = validate(args.path)
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    if errors:
        for e in errors:
            print("ERROR:", e)
        sys.exit(1)
    print("dump OK")


if __name__ == "__main__":
    main()
