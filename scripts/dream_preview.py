#!/usr/bin/env python
"""Render vivo: mostra o Dream-7B desmascarando código passe a passe.

  python scripts/dream_preview.py "Write def is_prime(n). Code only." \
      -n 128 -s 16 -t 8

Cada passe redesenha o buffer: posições ainda mascaradas aparecem como
░ e tokens resolvidos como texto — literalmente o preview de render.
"""
import argparse
import subprocess
import sys
import time

ENGINE = "/home/kron/Documents/projects/diffuse-cpp"
MODELS_DIR = "/home/kron/models/diffusion"
IMAGE = "localhost/llbuild:latest"
MASK_ID = 151666
END_IDS = {151643, 151645}

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(
            "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
    return _tok


def detokenize(ids, prompt_len):
    """Decode apenas a região gerada; máscaras -> ░."""
    tok = tokenizer()
    pieces = []
    run = []
    for t in ids[prompt_len:]:
        if t == MASK_ID:
            if run:
                pieces.append(tok.decode(run))
                run = []
            pieces.append("░")
        elif t in END_IDS:
            break
        else:
            run.append(t)
    if run:
        pieces.append(tok.decode(run))
    return "".join(pieces)


def render_preview(prompt, n, steps, threads, remasking, temp, seed):
    ids = tokenizer().apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_dict=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    ids_str = ",".join(map(str, ids))
    cmd = [
        "podman", "run", "--rm", "-i", "--network", "none",
        "-v", f"{ENGINE}:/src", "-v", f"{MODELS_DIR}:/models", "-w", "/src",
        IMAGE, "./build/preview-cli",
        "-m", "/models/dream-7b-q4km.gguf", "--tokens", ids_str,
        "-n", str(n), "-s", str(steps), "-t", str(threads),
        "--remasking", remasking, "--temp", str(temp), "--seed", str(seed),
        "--cache-keep-active", "1",
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True,
                            bufsize=1)
    plen = len(ids)
    print(f"\033[2J\033[H[render] '{prompt[:60]}' | s={steps} t={threads}\n")
    final = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("STEP "):
            hdr, _, body = line.partition(" ")
            step_tag, _, body = body.partition(" ")
            buf = [int(x) for x in body.split(",") if x]
            masked = sum(1 for t in buf[plen:] if t == MASK_ID)
            txt = detokenize(buf, plen)
            n_unm = len(buf[plen:]) - masked
            bar = "█" * (n_unm // 4) + "░" * (masked // 4)
            print(f"\033[H[render] passe {step_tag} | "
                  f"desmascarados {n_unm} restantes {masked} | "
                  f"{time.time()-t0:.0f}s\n{bar}\n\n{txt}\n")
        elif line.startswith("FINAL "):
            final = [int(x) for x in line[6:].split(",") if x]
    proc.wait()
    wall = time.time() - t0
    if final:
        txt = detokenize(final, plen)
        print(f"\n[render final] {wall:.0f}s\n{txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("-s", type=int, default=16)
    ap.add_argument("-t", type=int, default=8)
    ap.add_argument("--remasking", default="entropy_exit")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    render_preview(args.prompt, args.n, args.s, args.t, args.remasking,
                   args.temp, args.seed)


if __name__ == "__main__":
    main()
