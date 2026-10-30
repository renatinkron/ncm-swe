#!/usr/bin/env python3
"""Build the NCM training corpus (artifacts/corpus/).

Sources (idempotent per stage):
  seed      — PT-BR task seeds expanded to instruction/code records
  template  — parameterized PT-BR task prompts
  pylib     — real Python source harvested from local stdlib/site-packages
  qwen06b   — llama.cpp bulk generation (Qwen3-0.6B GGUF)
  qwen15b   — HF transformers generation (Qwen2.5-Coder-1.5B local dir)
  qwen30b   — llama.cpp capped generation (Qwen3-Coder-30B GGUF)

Output: artifacts/corpus/shard-*.jsonl + manifest.json
Record: {"id", "source", "text", "meta"}
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO / "artifacts" / "corpus"
LLAMA_CLI_DEFAULT = "/home/kron/.local/opt/llama-b10107/llama-cli"
MODEL_06B = "/home/kron/models/Qwen3-0.6B-Q8_0.gguf"
MODEL_30B = "/home/kron/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q3_K_XL.gguf"
HF_15B = "/tmp/llm/qwen15"

TEACHER_TPLS = [
    "Escreva uma função Python que {t}. Apenas código.",
    "Implemente em Python: {t}. Responda só com código.",
    "Explique em português como {t} e mostre um exemplo curto de código.",
    "Escreva um teste pytest para uma função que {t}.",
]

TASK_BANK = [
    "somar os elementos pares de uma lista", "inverter uma string sem slicing",
    "contar vogais em um texto", "verificar se um número é primo",
    "ordenar tuplas pelo segundo elemento", "remover duplicatas mantendo ordem",
    "calcular o fatorial", "converter Celsius para Fahrenheit",
    "encontrar o maior elemento", "achatar lista de listas",
    "contar palavras em uma frase", "verificar palíndromo",
    "gerar Fibonacci até N", "calcular média de uma lista",
    "validar CPF brasileiro", "implementar fila com duas pilhas",
    "busca binária", "mesclar dois dicionários",
    "encontrar segundo maior número", "codificar em rot13",
    "ler CSV e somar uma coluna", "implementar cache LRU",
    "converter decimal para binário", "verificar parênteses balanceados",
    "calcular juros compostos", "extrair domínio de e-mail",
    "gerar senha aleatória", "implementar paginação de lista",
    "avaliar expressão RPN", "agrupar anagramas",
    "classe para carrinho de compras", "contador de frequência de palavras",
    "sistema simples de notas de alunos", "conversor de unidades",
]


def hid(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]


# ------------------------------ sources ------------------------------------

def gen_seed_records(seeds_path: Path) -> list[dict]:
    out = []
    if not seeds_path.exists():
        return out
    for line in seeds_path.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        meta = {"level": s.get("level"), "seed_id": s.get("id")}
        out.append({"source": "seed", "meta": {**meta, "part": "solution"},
                    "text": f"# Tarefa: {s['prompt']}\n\n{s['solution']}"})
        out.append({"source": "seed", "meta": {**meta, "part": "instruct"},
                    "text": f"Instrução: {s['prompt']}\n\nResposta:\n{s['solution']}"})
        if s.get("test"):
            out.append({"source": "seed", "meta": {**meta, "part": "test"},
                        "text": f"{s['solution']}\n{s['test']}"})
    return out


def gen_template_records(n_variants: int = 4) -> list[dict]:
    out = []
    for task in TASK_BANK:
        for v in range(n_variants):
            out.append({"source": "template",
                        "meta": {"task": task, "variant": v},
                        "text": TEACHER_TPLS[v % len(TEACHER_TPLS)].format(t=task)})
    return out


def gen_pylib_records(roots: list[str], chunk: int, max_chars: int,
                      max_files: int) -> list[dict]:
    out, total = [], 0
    files = []
    for root in roots:
        for dirpath, _d, names in os.walk(root):
            for n in names:
                if n.endswith(".py"):
                    files.append(Path(dirpath) / n)
                    if len(files) >= max_files:
                        break
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break
    for p in files:
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#!"))
        for i in range(0, len(text), chunk):
            frag = text[i:i + chunk].strip()
            if len(frag) < 200:
                continue
            out.append({"source": "pylib", "meta": {"file": str(p), "ofs": i},
                        "text": frag})
            total += len(frag)
            if total >= max_chars:
                return out
    return out


# ------------------------------ teachers -----------------------------------

def _teacher_prompts() -> list[str]:
    return [t.format(t=task) for t in TEACHER_TPLS for task in TASK_BANK]


def _parse_llama_out(raw: str, prompt: str) -> str:
    out = raw
    idx = out.find(prompt)
    if idx >= 0:
        out = out[idx + len(prompt):]
    for marker in ("[ Prompt:", "Exiting"):
        out = out.split(marker)[0]
    for marker in ("[Start thinking]", "[End thinking]",
                   "[ Prompt:", "> "):
        out = out.replace(marker, "")
    return out.strip()


def _llama_gen(cli: str, model: str, prompt: str, n_tokens: int,
               threads: int, timeout: int) -> str | None:
    full = prompt + " /no_think"
    try:
        r = subprocess.run(
            [cli, "-m", model, "-p", full, "-n", str(n_tokens),
             "-t", str(threads), "--temp", "0.8", "--top-p", "0.95",
             "-cnv", "-st", "--no-display-prompt", "--no-warmup"],
            capture_output=True, text=True, timeout=timeout,
        )
        text = _parse_llama_out(r.stdout, full)
        return text if len(text) > 50 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def gen_teacher_llamacpp(source: str, cli: str, model: str, prompts: list[str],
                         max_seconds: int, workers: int, n_tokens: int,
                         threads: int) -> list[dict]:
    if not Path(cli).exists() or not Path(model).exists():
        return []
    out, deadline = [], time.time() + max_seconds
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        pending = list(prompts)
        while (pending or futs) and time.time() < deadline:
            while pending and len(futs) < workers and time.time() < deadline:
                p = pending.pop(0)
                futs[ex.submit(_llama_gen, cli, model, p, n_tokens,
                               threads, int(deadline - time.time()) + 30)] = p
            done = [f for f in futs if f.done()]
            for f in done:
                text = f.result()
                if text:
                    out.append({"source": source,
                                "meta": {"model": Path(model).name},
                                "text": text})
                del futs[f]
            if not done:
                time.sleep(0.3)
        for f in futs:
            f.cancel()
    return out


def gen_teacher_hf(model_dir: str, prompts: list[str], max_seconds: int,
                   n_tokens: int) -> list[dict]:
    if not Path(model_dir).exists():
        return []
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return []
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32)
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    out, deadline = [], time.time() + max_seconds
    for p in prompts:
        if time.time() >= deadline:
            break
        msgs = [{"role": "user", "content": p}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=n_tokens,
                                 do_sample=True, temperature=0.8, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()
        if len(text) > 50:
            out.append({"source": "qwen15b",
                        "meta": {"model": Path(model_dir).name}, "text": text})
    return out


# ------------------------------ writer -------------------------------------

def load_manifest(corpus_dir: Path) -> dict:
    p = corpus_dir / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sources": {}, "totals": {"records": 0, "chars": 0},
            "shards": [], "params": {}}


def write_records(corpus_dir: Path, manifest: dict, records: list[dict],
                  shard_size: int, dedup: bool = True) -> int:
    seen = set()
    if dedup:
        for sh in manifest["shards"]:
            for line in (corpus_dir / sh).read_text().splitlines():
                if line.strip():
                    seen.add(hid(json.loads(line)["text"]))
    n = 0
    buf = []
    shard_idx = len(manifest["shards"])
    for rec in records:
        key = hid(rec["text"])
        if dedup and key in seen:
            continue
        seen.add(key)
        rec = dict(rec)
        rec["id"] = f"{rec['source']}-{key}"
        buf.append(rec)
        if len(buf) >= shard_size:
            name = f"shard-{shard_idx:04d}.jsonl"
            with open(corpus_dir / name, "a") as f:
                for r in buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if name not in manifest["shards"]:
                manifest["shards"].append(name)
                shard_idx = len(manifest["shards"])
            n += len(buf)
            buf = []
    if buf:
        name = f"shard-{shard_idx:04d}.jsonl"
        with open(corpus_dir / name, "a") as f:
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if name not in manifest["shards"]:
            manifest["shards"].append(name)
        n += len(buf)
    return n


def stage_done(manifest: dict, source: str) -> bool:
    return manifest["sources"].get(source, {}).get("done", False)


def mark_stage(manifest: dict, source: str, n_records: int,
               n_chars: int) -> None:
    manifest["sources"][source] = {
        "done": True, "records": n_records, "chars": n_chars}
    manifest["totals"]["records"] += n_records
    manifest["totals"]["chars"] += n_chars


# ------------------------------ main ---------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR))
    ap.add_argument("--seeds", default=str(REPO / "data/seeds/seed_ptbr.jsonl"))
    ap.add_argument("--pylib-roots", nargs="*",
                    default=["/usr/lib/python3.13", "/usr/lib/python3.12",
                             "/tmp/llmenv/lib"])
    ap.add_argument("--pylib-chunk", type=int, default=1500)
    ap.add_argument("--pylib-max-chars", type=int, default=2_500_000)
    ap.add_argument("--pylib-max-files", type=int, default=4000)
    ap.add_argument("--llama-cli", default=LLAMA_CLI_DEFAULT)
    ap.add_argument("--model-06b", default=MODEL_06B)
    ap.add_argument("--model-30b", default=MODEL_30B)
    ap.add_argument("--hf-15b", default=HF_15B)
    ap.add_argument("--max-teacher-seconds", type=int, default=900)
    ap.add_argument("--teacher-n-tokens", type=int, default=280)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--sources", nargs="*",
                    default=["seed", "template", "pylib", "qwen06b", "qwen15b",
                             "qwen30b"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: no teachers, small pylib")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(corpus_dir)
    manifest["params"].update(vars(args))

    if args.smoke:
        args.sources = [s for s in args.sources
                        if s in ("seed", "template", "pylib")]
        args.pylib_max_chars = min(args.pylib_max_chars, 60_000)
        args.pylib_max_files = min(args.pylib_max_files, 60)

    generators = {
        "seed": lambda: gen_seed_records(Path(args.seeds)),
        "template": lambda: gen_template_records(),
        "pylib": lambda: gen_pylib_records(
            args.pylib_roots, args.pylib_chunk, args.pylib_max_chars,
            args.pylib_max_files),
        "qwen06b": lambda: gen_teacher_llamacpp(
            "qwen06b", args.llama_cli, args.model_06b, _teacher_prompts(),
            args.max_teacher_seconds, args.workers, args.teacher_n_tokens,
            args.threads),
        "qwen15b": lambda: gen_teacher_hf(
            args.hf_15b, _teacher_prompts(), args.max_teacher_seconds,
            args.teacher_n_tokens),
        "qwen30b": lambda: gen_teacher_llamacpp(
            "qwen30b", args.llama_cli, args.model_30b, _teacher_prompts()[:40],
            args.max_teacher_seconds, 1, args.teacher_n_tokens, 8),
    }

    for src in args.sources:
        if src not in generators or stage_done(manifest, src):
            print(f"[corpus] skip {src} (done)", flush=True)
            continue
        t0 = time.time()
        recs = generators[src]()
        n = write_records(corpus_dir, manifest, recs, args.shard_size)
        chars = sum(len(r["text"]) for r in recs[:n]) if n else 0
        mark_stage(manifest, src, n, chars)
        (corpus_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"[corpus] {src}: +{n} records ({chars} chars) "
              f"in {time.time()-t0:.0f}s", flush=True)

    print(json.dumps(manifest["totals"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
