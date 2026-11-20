#!/usr/bin/env python
"""Painel interativo do renderizador de código por difusão (Dream-7B).

  python scripts/dream_panel.py

Digite uma intenção -> o código se materializa passe a passe na tela,
como o preview de um ray-tracer. Ajuste os knobs entre renders:

  :steps 16        passes máximos de denoising (budget de render)
  :n 160           tokens do canvas
  :temp 0.3        temperatura (<=0.3 recomendado pra código)
  :remasking X     entropy_exit | low_confidence | random | maskgit_plus
  :threads 8       threads do forward
  :seed 42         seed (-1 = aleatório)
  :preview off     desliga o render vivo (só resultado final)
  :eval            alterna checagem AST do código final (on/off)
  :knobs           mostra config atual
  :quit            sai
"""
import ast
import random
import re
import time

from dream_preview import detokenize, tokenizer
import dream_preview as dp

MASK_ID = dp.MASK_ID


def render(prompt, cfg):
    ids = tokenizer().apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_dict=False)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    ids_str = ",".join(map(str, ids))
    import subprocess
    cmd = [
        "podman", "run", "--rm", "-i", "--network", "none",
        "-v", f"{dp.ENGINE}:/src", "-v", f"{dp.MODELS_DIR}:/models",
        "-w", "/src", dp.IMAGE, "./build/preview-cli",
        "-m", "/models/dream-7b-q4km.gguf", "--tokens", ids_str,
        "-n", str(cfg["n"]), "-s", str(cfg["steps"]),
        "-t", str(cfg["threads"]), "--remasking", cfg["remasking"],
        "--temp", str(cfg["temp"]), "--seed", str(cfg["seed"]),
        "--cache-keep-active", "1",
    ]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    plen = len(ids)
    if cfg["preview"]:
        print(f"\033[2J\033[H[render] '{prompt[:50]}' | s={cfg['steps']} "
              f"T={cfg['temp']} {cfg['remasking']}\n")
    final = None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("STEP ") and cfg["preview"]:
            hdr, _, body = line.partition(" ")
            step_tag, _, body = body.partition(" ")
            buf = [int(x) for x in body.split(",") if x]
            masked = sum(1 for t in buf[plen:] if t == MASK_ID)
            txt = detokenize(buf, plen)
            n_unm = len(buf[plen:]) - masked
            bar = "█" * (n_unm // 4) + "░" * (masked // 4)
            print(f"\033[H[render] passe {step_tag} | desmascarados "
                  f"{n_unm} restantes {masked} | "
                  f"{time.time()-t0:.0f}s\n{bar}\n\n{txt}\n")
        elif line.startswith("FINAL "):
            final = [int(x) for x in line[6:].split(",") if x]
    proc.wait()
    wall = time.time() - t0
    if not final:
        print("[render] sem saída do engine")
        return
    txt = detokenize(final, plen)
    if cfg["preview"]:
        print(f"\n[final] {wall:.0f}s\n{txt}")
    else:
        print(f"[final] {wall:.0f}s\n{txt}")
    if cfg["eval"]:
        m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S)
        code = (m.group(1) if m else txt).strip()
        try:
            ast.parse(code)
            print("[eval] AST OK")
        except SyntaxError as e:
            print(f"[eval] AST-FAIL: {e}")


def main():
    cfg = {"steps": 16, "n": 160, "temp": 0.3, "remasking": "entropy_exit",
           "threads": 8, "seed": -1, "preview": True, "eval": True}
    print(__doc__.strip().splitlines()[0])
    print("intenção = renderiza | :knobs mostra config | :quit sai")
    while True:
        try:
            line = input("\ndream> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith(":"):
            parts = line[1:].split()
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else None
            if cmd in ("quit", "q", "exit"):
                break
            elif cmd == "knobs":
                print(cfg)
            elif cmd == "preview":
                cfg["preview"] = (arg != "off")
            elif cmd == "eval":
                cfg["eval"] = (arg != "off") if arg else not cfg["eval"]
            elif cmd in cfg and arg is not None:
                try:
                    cfg[cmd] = type(cfg[cmd])(arg)
                except (ValueError, TypeError):
                    cfg[cmd] = arg
            else:
                print("comando desconhecido")
            continue
        cfg_run = dict(cfg)
        if cfg_run["seed"] == -1:
            cfg_run["seed"] = random.randint(0, 2**31 - 1)
        render(line, cfg_run)


if __name__ == "__main__":
    main()
