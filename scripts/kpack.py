#!/usr/bin/env python3
"""RUN-036 T4: portable knowledge pack + retrieval transplant test.

Builds a model-agnostic knowledge catalog from the convdistill transcript:
each lesson becomes a card {skill, probe, ideal_answer, expected_tokens,
round}. Cards + embedding index = "knowledge pack" any model can query by
niche instead of holding the knowledge in weights.

Transplant test: the (trained) student answers held-out eval questions
(a) naked vs (b) with top-2 retrieved cards injected in the prompt.
catalog > naked proves the pack delivers knowledge the weights don't retain.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("THREADS", 6)))

ART = Path("artifacts/convdistill")
STUDENT_DIR = os.environ.get("STUDENT", os.path.expanduser("~/models/qwen05-sft"))
MAX_NEW = int(os.environ.get("MAX_NEW", 200))
TOPK = int(os.environ.get("TOPK", 2))

LT = chr(60) + "|"
IMS = LT + "im_start" + chr(62)
IME = LT + "im_end" + chr(62)


def render(user_msg, system="You are a student learning to code. Answer briefly and exactly."):
    return (IMS + "system\n" + system + IME + "\n" +
            IMS + "user\n" + user_msg + IME + "\n" + IMS + "assistant\n")


def embed_text(model, tok, text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids
    with torch.no_grad():
        emb = model.model.embed_tokens(ids).mean(1)[0]
    return torch.nn.functional.normalize(emb.float(), dim=0)


def gen(model, tok, prompt):
    ids = tok(render(prompt), return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=tok.convert_tokens_to_ids(IME),
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def sig_rubric(ideal):
    """Deterministic expected tokens from the teacher's frozen answer.
    Identifiers only — generic keywords saturate (any Python answer hits them)."""
    return sorted(set(re.findall(r"(?:def|class)\s+(\w+)", ideal)))


def rubric(attempt, expected):
    if not expected:
        return 0.0
    return sum(1 for t in expected if t in attempt) / len(expected)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    transcript = [json.loads(l) for l in open(ART / "transcript.jsonl")
                  if l.strip() and "teacher" in json.loads(l)]
    cards = [{
        "card_id": i,
        "skill": r["teacher"]["skill"],
        "probe": r["teacher"]["probe"],
        "ideal_answer": r["teacher"]["ideal_answer"],
        "expected_tokens": r["teacher"]["expected_tokens"],
        "strategy": r["teacher"]["strategy_note"],
        "round": r["round"],
        "format": "kpack.v1",
    } for i, r in enumerate(transcript)]

    tok = AutoTokenizer.from_pretrained(STUDENT_DIR)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_DIR, dtype=torch.float32)
    sd = ART / "student_final.pt"
    weights_used = "base"
    if sd.exists():
        model.load_state_dict(torch.load(sd, map_location="cpu"))
        weights_used = "trained"
        print("loaded trained student", flush=True)
    else:
        print("WARNING: student_final.pt missing — evaluating BASE weights", flush=True)
    model.eval()

    # index: embed "skill + probe" — the retrieval key is the task, not the answer
    keys = [c["skill"] + " :: " + c["probe"] for c in cards]
    index = torch.stack([embed_text(model, tok, k) for k in keys])
    import numpy as np
    np.save(ART / "kpack_index.npy", index.numpy())

    pack = {
        "format": "kpack.v1",
        "description": "portable condensed-knowledge catalog: cards readable by "
                       "any model; index embeds the task (skill+probe), not the "
                       "answer. Query by cosine over your own embedding space.",
        "n_cards": len(cards),
        "embedding": {"model": "qwen2.5-0.5b embed_tokens mean-pool", "dim": 896},
        "cards": cards,
    }
    (ART / "knowledge_pack.json").write_text(
        json.dumps(pack, ensure_ascii=False, indent=1))
    print(f"knowledge_pack: {len(cards)} cards", flush=True)

    frozen = json.loads((ART / "eval_frozen.json").read_text())
    results = []
    for q in frozen:
        question, ideal = q["question"], q["ideal_answer"]
        expected = sig_rubric(ideal)
        naked = gen(model, tok, question)
        qv = embed_text(model, tok, question)
        sims = index @ qv
        top = sims.topk(min(TOPK, len(cards))).indices.tolist()
        refs = "\n\n".join(
            f"--- known answer for: {cards[i]['skill']} ---\n"
            f"{cards[i]['ideal_answer']}" for i in top)
        armed = gen(model, tok,
                    "Reference knowledge (use if relevant):\n" + refs +
                    "\n\nTask: " + question)
        res = {
            "question": question[:80],
            "naked_rubric": round(rubric(naked, expected), 3),
            "armed_rubric": round(rubric(armed, expected), 3),
            "retrieved": [cards[i]["skill"] for i in top],
        }
        results.append(res)
        print(f"  naked={res['naked_rubric']:.2f} armed={res['armed_rubric']:.2f} "
              f"via {res['retrieved']}", flush=True)

    out = {
        "n": len(results), "weights": weights_used,
        "naked_mean": round(sum(r["naked_rubric"] for r in results) / len(results), 3),
        "armed_mean": round(sum(r["armed_rubric"] for r in results) / len(results), 3),
        "per_question": results,
    }
    (ART / "kpack_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "naked_mean", "armed_mean")}))


if __name__ == "__main__":
    main()
