#!/usr/bin/env python3
"""Interactive chat with the RUN-036 trained student (0.5B).

Persistent REPL: one user line in -> one reply out. Each exchange appended to
artifacts/convdistill/livechat.jsonl so the conversation can be tailed live.
Usage: ~/.venvs/ncm/bin/python scripts/chat_student.py
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("THREADS", 6)))

ART = Path("artifacts/convdistill")
SD = os.environ.get("STUDENT", os.path.expanduser("~/models/qwen05-sft"))
CKPT = ART / "student_final.pt"
MAX_NEW = int(os.environ.get("MAX_NEW", 200))
TEMP = float(os.environ.get("TEMP", 0.7))

LT = chr(60) + "|"
IMS = LT + "im_start" + chr(62)
IME = LT + "im_end" + chr(62)


def render(history, user_msg):
    text = IMS + "system\nYou are a student learning to code. Answer briefly and exactly." + IME + "\n"
    for u, a in history[-6:]:
        text += IMS + "user\n" + u + IME + "\n" + IMS + "assistant\n" + a + IME + "\n"
    return text + IMS + "user\n" + user_msg + IME + "\n" + IMS + "assistant\n"


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SD)
    model = AutoModelForCausalLM.from_pretrained(SD, dtype=torch.float32)
    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        print(f"[loaded trained student {CKPT}]", flush=True)
    model.eval()
    eos = tok.convert_tokens_to_ids(IME)
    history = []
    print("READY — type a message, ctrl-d to quit", flush=True)
    for line in sys.stdin:
        msg = line.strip()
        if not msg:
            continue
        ids = tok(render(history, msg), return_tensors="pt").input_ids
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=(TEMP > 0),
                temperature=TEMP if TEMP > 0 else None,
                top_p=0.9 if TEMP > 0 else None,
                eos_token_id=eos, pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        history.append((msg, reply))
        with open(ART / "livechat.jsonl", "a") as f:
            f.write(json.dumps({"t": time.time(), "user": msg,
                                "reply": reply}, ensure_ascii=False) + "\n")
        print(f"REPLY({time.time()-t0:.1f}s)> {reply}", flush=True)


if __name__ == "__main__":
    main()
