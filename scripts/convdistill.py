#!/usr/bin/env python3
"""RUN-036: conversational distillation with meta-teaching.

Teacher (30B via llama-server OpenAI API) adaptively probes and teaches a
0.5B student (HF checkpoint) which trains online (SFT per round). The teacher
sees per-round learning deltas and learns which teaching strategy retains.
Artifacts: artifacts/convdistill/{metrics,transcript}.jsonl, eval_frozen.json,
student_final.pt. Original student checkpoint is never modified.
"""

import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import torch

STUDENT_DIR = os.environ.get("STUDENT", os.path.expanduser("~/models/qwen05-sft"))
PORT = int(os.environ.get("PORT", 8089))
ART = Path("artifacts/convdistill")
ROUNDS = int(os.environ.get("ROUNDS", 24))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 6))
LRV = os.environ.get("LR")
MODE = os.environ.get("MODE", "auto")  # auto|full|lora
MAX_NEW = int(os.environ.get("MAX_NEW", 160))
THREADS = int(os.environ.get("THREADS", 8))
LORA_R = int(os.environ.get("LORA_R", 8))

torch.set_num_threads(THREADS)

TEACHER_SYS = """You are teaching a 0.5B-parameter student language model to code Python.
Each round you design ONE probe/task, give your ideal answer, and list check tokens.
Your goal: maximize the student's measurable learning per round. You will get feedback
on which of your past strategies produced retention (student re-answering a previously
failed item correctly after ONE training step). Adapt difficulty, answer length, and
style accordingly. The student is tiny: teach ONE small thing per round, short answers.

Rules:
- probe: a small task (one function or one concept), self-contained.
- ideal_answer: the exact answer you want the student to learn. SHORT (<15 lines),
  plain code or code+1-line note. Put the answer in the format you want the student
  to imitate (it is trained verbatim on this text).
- expected_tokens: 4-8 short strings that MUST appear verbatim in your ideal_answer
  and would appear in any correct answer (identifiers, keywords, output values).
- strategy_note: 1 line on what you are trying this round and why.
Output STRICT JSON only, no markdown fences:
{"skill": "...", "probe": "...", "ideal_answer": "...", "expected_tokens": ["..."], "strategy_note": "..."}"""

EVAL_QUESTIONS = [
    "Write a Python function is_palindrome(s) that checks whether a string reads the same forwards and backwards.",
    "Write a Python function second_largest(xs) returning the second largest distinct value in a list, or None.",
    "Write a Python function count_vowels(s) returning the number of vowels in a string.",
    "Write a Python function merge_dicts(a, b) returning a new dict where b's keys override a's.",
    "Write a Python function fib(n) returning the n-th Fibonacci number iteratively (fib(0)=0, fib(1)=1).",
    "Write a Python function flatten(nested) that flattens a list of lists one level deep.",
    "Write a Python class Stack with push, pop, and is_empty methods backed by a list.",
    "Write a Python function char_freq(s) returning a dict of character frequencies.",
]

NEUTRAL_TEXT = (
    "The library was quiet in the late afternoon. Outside, rain moved across "
    "the courtyard in soft waves, and the windows fogged at the corners. A "
    "student returned a book to the wrong shelf, then laughed and corrected "
    "the mistake. On the table lay a notebook full of small sketches of birds, "
    "each labeled with a date and a place. The clock above the door ticked "
    "steadily, and for a while nothing at all was urgent."
)


def jload(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def teacher_chat(messages, temperature=0.3, max_tokens=700):
    body = json.dumps({
        "model": "teacher",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], time.time() - t0


def teacher_health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def parse_teacher_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    need = {"skill", "probe", "ideal_answer", "expected_tokens", "strategy_note"}
    if not need.issubset(d) or not isinstance(d["expected_tokens"], list):
        return None
    return d


def rubric(attempt, expected_tokens):
    if not expected_tokens:
        return 0.0
    hits = sum(1 for t in expected_tokens if t in attempt)
    return hits / len(expected_tokens)


class LoRA(torch.nn.Module):
    def __init__(self, linear, r):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        dev, dt = linear.weight.device, linear.weight.dtype
        self.A = torch.nn.Parameter(torch.zeros(r, linear.in_features, device=dev, dtype=dt))
        self.B = torch.nn.Parameter(torch.zeros(linear.out_features, r, device=dev, dtype=dt))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = 16.0 / r

    def forward(self, x):
        return self.linear(x) + (x @ self.A.t()) @ self.B.t() * self.scale


def inject_lora(model, r):
    names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    n = 0
    for mod in model.modules():
        for name in names:
            if hasattr(mod, name) and isinstance(getattr(mod, name), torch.nn.Linear):
                setattr(mod, name, LoRA(getattr(mod, name), r))
                n += 1
    return n


def merge_lora(model):
    """Fold LoRA deltas back into base Linear weights and unwrap modules."""
    n = 0
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRA):
                with torch.no_grad():
                    child.linear.weight += (child.B @ child.A) * child.scale
                setattr(mod, name, child.linear)
                n += 1
    return n

# special tokens (built without literal seq for editor safety)
LT = chr(60) + "|"
IMS = LT + "im_start" + chr(62)
IME = LT + "im_end" + chr(62)
SYS_OPEN = IMS + "system" + chr(10)
USER_OPEN = IMS + "user" + chr(10)
ASST_OPEN = IMS + "assistant" + chr(10)
IME_NL = IME + chr(10)

STUDENT_SYS = "You are a student learning to code. Answer briefly and exactly."


def render_chat(user_msg, system=STUDENT_SYS):
    return SYS_OPEN + system + IME_NL + USER_OPEN + user_msg + IME_NL + ASST_OPEN


def student_gen(model, tok, prompt, max_new=MAX_NEW):
    ids = tok(render_chat(prompt), return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids(IME),
                             pad_token_id=tok.eos_token_id)
    new = out[0][ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), time.time() - t0


def ce_on_text(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids
    if ids.shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        logits = model(ids).logits[0, :-1]
        labels = ids[0, 1:]
        return torch.nn.functional.cross_entropy(logits, labels).item()


def sft_step(model, tok, opt, question, answer):
    prompt = render_chat(question)
    full = prompt + answer + IME_NL
    pid = tok(prompt, return_tensors="pt").input_ids
    fid = tok(full, return_tensors="pt").input_ids
    labels = fid.clone()
    labels[:, : pid.shape[1]] = -100
    if (labels != -100).sum().item() < 2:
        return float("nan"), 0
    model.train()
    logits = model(fid).logits
    loss = torch.nn.functional.cross_entropy(
        logits[0, :-1], labels[0, 1:], ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    model.eval()
    return loss.item(), int((labels != -100).sum().item())


def eval_suite(model, tok, frozen, snap, log_prefix):
    ces = [ce_on_text(model, tok, q["ideal_answer"]) for q in frozen]
    ppl = math.exp(ce_on_text(model, tok, NEUTRAL_TEXT))
    drift = {}
    for name, w0 in snap.items():
        cur = dict(model.named_parameters())[name].data
        drift[name] = round((cur.float() - w0.float()).norm().item(), 4)
    ce_mean = sum(ces) / len(ces)
    print(f"{log_prefix} ce_teacher={ce_mean:.4f} ppl_neutral={ppl:.2f} "
          f"drift={drift}", flush=True)
    return {"ce_teacher": ce_mean, "ppl_neutral": ppl, "weight_drift": drift}


def main():
    ART.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    metrics_f, transcript_f = ART / "metrics.jsonl", ART / "transcript.jsonl"

    if not teacher_health():
        sys.exit(f"teacher not reachable on :{PORT} — start llama-server first")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT_DIR)

    avail_gb = float(re.search(r"MemAvailable:\s+(\d+)",
                               Path("/proc/meminfo").read_text()).group(1)) / 1e6
    mode = MODE
    if mode == "auto":
        mode = "full" if avail_gb >= 14 else "lora"
    lr = float(LRV) if LRV else (1e-5 if mode == "full" else 2e-4)
    print(f"mode={mode} lr={lr} avail={avail_gb:.1f}GB threads={THREADS}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.float32)
    if mode == "lora":
        n_lora = inject_lora(model, LORA_R)
        for p in model.parameters():
            p.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, LoRA):
                for p in m.parameters(recurse=False):
                    if p is not m.linear.weight:
                        p.requires_grad_(True)
        print(f"lora adapters injected: {n_lora}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    # snapshot for weight drift
    snap = {}
    for name, p in model.named_parameters():
        if p.requires_grad and len(snap) < 4:
            snap[name] = p.data.clone()
        if len(snap) >= 4:
            break

    # T2: frozen teacher eval set (generated BEFORE any training)
    frozen_path = ART / "eval_frozen.json"
    frozen = jload(frozen_path)
    if frozen is None:
        print("generating frozen teacher eval set (pre-training)...", flush=True)
        frozen = []
        for q in EVAL_QUESTIONS:
            ans, _ = teacher_chat(
                [{"role": "user", "content":
                  "Answer briefly with clean code only:\n\n" + q}],
                temperature=0.2, max_tokens=400)
            frozen.append({"question": q, "ideal_answer": ans})
        frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=1))

    history, failed_items = [], []
    t_start = time.time()

    # baseline eval BEFORE first SFT step
    base = eval_suite(model, tok, frozen, snap, "baseline")
    append_jsonl(metrics_f, {"round": -1, "kind": "baseline", **base})

    for rnd in range(ROUNDS):
        t_r = time.time()
        feedback = json.dumps({"recent": history[-8:]}, ensure_ascii=False)
        msg = (f"Round {rnd}. Feedback on your teaching so far (pre_pass = student "
               f"correct before training; retention = re-answered a failed item "
               f"correctly after 1 SFT step):\n{feedback}\n\n"
               f"Design this round's probe. Escalate or pivot based on what retained.")
        if rnd == 0:
            msg = ("Round 0. No history yet — start with a basic diagnostic probe "
                   "(simple function writing). Output the JSON.")
        raw, teacher_dt = None, 0.0
        for attempt in range(2):
            text, dt = teacher_chat(
                [{"role": "system", "content": TEACHER_SYS},
                 {"role": "user", "content": msg}])
            teacher_dt += dt
            raw = parse_teacher_json(text)
            if raw:
                break
            msg = "Your last reply was not valid JSON. Output ONLY the JSON object."
        if not raw:
            append_jsonl(transcript_f, {"round": rnd, "error": "teacher_json_parse"})
            continue

        # 1) student pre-test
        attempt_text, sg_dt = student_gen(model, tok, raw["probe"])
        pre = rubric(attempt_text, raw["expected_tokens"])

        # 2) SFT step on teacher's ideal answer
        loss, n_tok = sft_step(model, tok, opt, raw["probe"], raw["ideal_answer"])

        # 3) retention probe: re-answer most recent failed item
        ret, re_att = None, None
        if failed_items:
            fi = failed_items[-1]
            re_att, _ = student_gen(model, tok, fi["probe"], max_new=MAX_NEW)
            ret = rubric(re_att, fi["expected_tokens"])
            fi["retries"] += 1
            if ret >= 0.75:
                failed_items.remove(fi)

        passed = pre >= 0.75
        if not passed:
            failed_items.append(
                {"probe": raw["probe"], "expected_tokens": raw["expected_tokens"],
                 "retries": 0})
            if len(failed_items) > 20:
                failed_items.pop(0)

        rec = {
            "round": rnd, "skill": raw["skill"], "strategy_note": raw["strategy_note"],
            "pre_pass": passed, "pre_rubric": round(pre, 3),
            "retention": (round(ret, 3) if ret is not None else None),
            "sft_loss": (round(loss, 4) if math.isfinite(loss) else None),
            "sft_tokens": n_tok,
            "sec_teacher": round(teacher_dt, 1), "sec_student": round(sg_dt, 1),
            "sec_round": round(time.time() - t_r, 1),
        }
        append_jsonl(metrics_f, rec)
        append_jsonl(transcript_f, {
            "round": rnd, "teacher": raw, "student_attempt": attempt_text,
            "retention_attempt": re_att})
        history.append({"skill": raw["skill"], "strategy": raw["strategy_note"],
                        "pre_pass": passed, "retention": ret})
        print(f"r{rnd:02d} skill={raw['skill'][:28]:28s} pre={pre:.2f} "
              f"ret={ret if ret is not None else '-'} loss={loss:.3f} "
              f"t={rec['sec_round']}s", flush=True)

        if (rnd + 1) % EVAL_EVERY == 0:
            ev = eval_suite(model, tok, frozen, snap, f"eval@{rnd+1}")
            append_jsonl(metrics_f, {"round": rnd, "kind": "eval", **ev})
            torch.save(model.state_dict(), ART / "student_ckpt.pt")

    final = eval_suite(model, tok, frozen, snap, "final")
    if mode == "lora":
        print(f"merging {merge_lora(model)} lora adapters before save", flush=True)
    torch.save(model.state_dict(), ART / "student_final.pt")
    append_jsonl(metrics_f, {
        "round": ROUNDS, "kind": "final", "mode": mode, "lr": lr, "seed": 0,
        "wall_min": round((time.time() - t_start) / 60, 1), **final})
    print(f"done: mode={mode} wall={round((time.time()-t_start)/60,1)}min", flush=True)


if __name__ == "__main__":
    main()
