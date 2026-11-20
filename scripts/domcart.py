#!/usr/bin/env python3
"""RUN-037: pluggable domain cartridges.

Generalist 0.5B + LoRA-per-domain "cartridge" (few MB) trained on teacher
lessons. Proof: plugged -> specialist on domain eval (regex EXECUTED against
test cases); unplugged -> generalist intact (pass + neutral PPL unchanged).
Domain: Python regex.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from convdistill import (LoRA, merge_lora, inject_lora, render_chat,
                         teacher_chat, teacher_health, parse_teacher_json,
                         ce_on_text, NEUTRAL_TEXT)

torch.set_num_threads(int(os.environ.get("THREADS", 6)))

STUDENT_DIR = os.environ.get("STUDENT", os.path.expanduser("~/models/qwen05-sft"))
ART = Path("artifacts/domcart")
LESSONS = int(os.environ.get("LESSONS", 24))
LORA_R = int(os.environ.get("LORA_R", 8))
LR = float(os.environ.get("LR", 2e-4))
MAX_NEW = int(os.environ.get("MAX_NEW", 200))

DOM_SYS = """You create tiny Python-regex lessons for a 0.5B student model.
Each lesson: one focused probe + its ideal short answer + check tokens.
Cover distinct regex skills: character classes, anchors, quantifiers,
groups/capture, alternation, escaping, re.search vs fullmatch, findall, sub.
Answers SHORT (<10 lines): the regex pattern as a raw string plus optionally
2 lines of usage. Output STRICT JSON only:
{"skill": "...", "probe": "...", "ideal_answer": "...", "expected_tokens": ["..."], "strategy_note": "..."}"""

EVAL_TASKS = [
    {"task": "Write a regex that fully matches a simple email like user@site.com.",
     "tests": [("user@site.com", True), ("a-b@c.d", True), ("not an email", False),
                ("user@site", False)]},
    {"task": "Write a regex that fully matches a Brazilian CEP like 12345-678.",
     "tests": [("12345-678", True), ("00000-000", True), ("12345678", False),
                ("1234-5678", False)]},
    {"task": "Write a regex that fully matches a date in dd/mm/yyyy format.",
     "tests": [("25/12/2023", True), ("01/01/2000", True), ("2023-12-25", False),
                ("5/3/99", False)]},
    {"task": "Write a regex that fully matches an integer (optional sign).",
     "tests": [("42", True), ("-17", True), ("+9", True), ("4.5", False),
                ("abc", False)]},
    {"task": "Write a regex that fully matches a hex color like #a1B2c3.",
     "tests": [("#a1B2c3", True), ("#FFFFFF", True), ("#123", False),
                ("a1B2c3", False)]},
    {"task": "Write a regex that fully matches a snake_case identifier.",
     "tests": [("my_var_2", True), ("snake_case", True), ("MyVar", False),
                ("2var", False), ("__x", False)]},
    {"task": "Write a regex that fully matches a phone like (11) 98765-4321.",
     "tests": [("(11) 98765-4321", True), ("(00) 00000-0000", True),
                ("11 98765-4321", False), ("(11) 8765-4321", False)]},
    {"task": "Write a regex that fully matches a word starting with an uppercase letter.",
     "tests": [("Hello", True), ("World9", True), ("hello", False),
                ("9Hello", False)]},
]

PATTERN_RE = re.compile(
    r"r\"([^\"\n]+)\"|r'([^'\n]+)'|re\.compile\(\s*[\"']([^\"']+)[\"']|"
    r"[\"'`](\\\\d[^\"'`\n]{2,}?|[\\w^\[\]().*+?{}$|\-]{4,}?)[\"'`]")


def extract_pattern(answer):
    for m in PATTERN_RE.finditer(answer):
        for g in m.groups():
            if g:
                return g
    stripped = answer.strip().split("\n")[0].strip().strip('`"\'')
    return stripped if len(stripped) < 120 else None


TEST_WORKER = """
import json, re, sys
data = json.loads(sys.argv[1])
try:
    p = re.compile(data["pattern"])
except Exception as e:
    print(json.dumps({"ok": False, "err": "compile"}))
    sys.exit()
res = []
for s, want in data["tests"]:
    got = bool(p.fullmatch(s))
    res.append(got == want)
print(json.dumps({"ok": True, "passed": sum(res), "total": len(res)}))
"""


def run_pattern_tests(pattern, tests, timeout=5):
    try:
        out = subprocess.run(
            [sys.executable, "-c", TEST_WORKER,
             json.dumps({"pattern": pattern, "tests": tests})],
            capture_output=True, text=True, timeout=timeout)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "err": str(e)[:60]}


def student_answer(model, tok, prompt):
    ids = tok(render_chat(prompt), return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids(
                                 chr(60) + "|im_end" + chr(62)),
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def eval_domain(model, tok, tag):
    rows, passed, total = [], 0, 0
    for it in EVAL_TASKS:
        ans = student_answer(model, tok, it["task"])
        pat = extract_pattern(ans)
        res = run_pattern_tests(pat, it["tests"]) if pat else {"ok": False, "err": "no_pattern"}
        p = res.get("passed", 0)
        rows.append({"task": it["task"][:60], "pattern": pat,
                     "passed": p, "total": res.get("total", len(it["tests"])),
                     "compile": res.get("ok", False)})
        passed += p
        total += res.get("total", len(it["tests"]))
        print(f"  [{tag}] {p}/{res.get('total')} pat={str(pat)[:50]!r}", flush=True)
    rate = passed / total if total else 0.0
    return {"pass_rate": round(rate, 3), "passed": passed, "total": total,
            "rows": rows}


def unwrap_lora(model):
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRA):
                setattr(mod, name, child.linear)


def lora_state(model):
    sd = {}
    for mname, mod in model.named_modules():
        for cname, child in mod.named_children():
            if isinstance(child, LoRA):
                sd[f"{mname}.{cname}"] = {"A": child.A.data.clone(),
                                          "B": child.B.data.clone()}
    return sd


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ART.mkdir(parents=True, exist_ok=True)
    metrics_f = ART / "metrics.jsonl"
    torch.manual_seed(0)

    if not teacher_health():
        sys.exit("teacher not reachable on :8089")

    tok = AutoTokenizer.from_pretrained(STUDENT_DIR)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.float32)
    model.eval()

    # --- 1) baseline eval (generalist, no cartridge) ---
    print("baseline eval (unplugged)...", flush=True)
    base = eval_domain(model, tok, "base")
    base_ppl = math.exp(ce_on_text(model, tok, NEUTRAL_TEXT))
    print(f"BASE pass_rate={base['pass_rate']} ppl={base_ppl:.2f}", flush=True)

    # --- 2) teacher generates domain lessons ---
    lessons, covered = [], []
    for i in range(LESSONS):
        msg = (f"Lesson {i+1}/{LESSONS}. Already covered skills: {covered}. "
               "Pick an UNCOVERED regex skill. Output the JSON.")
        raw = None
        for _ in range(2):
            try:
                txt, _ = teacher_chat(
                    [{"role": "system", "content": DOM_SYS},
                     {"role": "user", "content": msg}])
                raw = parse_teacher_json(txt)
            except Exception as e:
                print("teacher err:", e, flush=True)
            if raw:
                break
            msg = "Invalid JSON. Output ONLY the JSON object."
        if not raw:
            continue
        lessons.append(raw)
        covered.append(raw["skill"])
        print(f"lesson {i+1}: {raw['skill']}", flush=True)
    (ART / "pack_regex.json").write_text(
        json.dumps({"format": "kpack.v1", "domain": "python-regex",
                    "cards": lessons}, ensure_ascii=False, indent=1))

    # --- 3) LoRA cartridge training (base frozen) ---
    n_inj = inject_lora(model, LORA_R)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRA):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
    tr = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in tr)
    print(f"lora injected={n_inj} trainable={n_tr/1e6:.2f}M lr={LR}", flush=True)
    opt = torch.optim.AdamW(tr, lr=LR, weight_decay=0.0)
    from convdistill import sft_step
    for epoch in range(2):
        for i, L in enumerate(lessons):
            loss, nt = sft_step(model, tok, opt, L["probe"], L["ideal_answer"])
            if epoch == 1:
                print(f"  e1 l{i} loss={loss:.3f}", flush=True)
    model.eval()

    # --- 4) plugged eval ---
    print("plugged eval...", flush=True)
    plug = eval_domain(model, tok, "plug")
    plug_ppl = math.exp(ce_on_text(model, tok, NEUTRAL_TEXT))

    # --- 5) save cartridge (adapters only — the "download") ---
    cart_path = ART / "cartridge_regex.pt"
    torch.save({"lora_r": LORA_R, "domain": "python-regex",
                "state": lora_state(model)}, cart_path)
    cart_bytes = cart_path.stat().st_size

    # --- 6) unplugged eval (cart removed) ---
    unwrap_lora(model)
    model.eval()
    print("unplugged eval...", flush=True)
    unplug = eval_domain(model, tok, "unplug")
    unplug_ppl = math.exp(ce_on_text(model, tok, NEUTRAL_TEXT))

    rec = {
        "n_tasks": len(EVAL_TASKS), "lessons": len(lessons),
        "lora_params_M": round(n_tr / 1e6, 2), "cartridge_bytes": cart_bytes,
        "base_pass_rate": base["pass_rate"], "plug_pass_rate": plug["pass_rate"],
        "unplug_pass_rate": unplug["pass_rate"],
        "base_ppl": round(base_ppl, 2), "plug_ppl": round(plug_ppl, 2),
        "unplug_ppl": round(unplug_ppl, 2),
        "base_rows": base["rows"], "plug_rows": plug["rows"],
        "unplug_rows": unplug["rows"],
    }
    with open(metrics_f, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({k: rec[k] for k in (
        "base_pass_rate", "plug_pass_rate", "unplug_pass_rate",
        "cartridge_bytes", "lora_params_M")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
