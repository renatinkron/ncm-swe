import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _tiny_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with open(corpus / "shard-0000.jsonl", "w") as f:
        for i in range(40):
            f.write(json.dumps(
                {"id": f"seed-{i}", "source": "seed",
                 "text": f"def fn{i}(x): return x * {i}  # comentario {i}"},
                ensure_ascii=False) + "\n")
    return corpus


def test_vqvae_train_resume(tmp_path):
    ckpt = tmp_path / "vqvae.pt"
    out = tmp_path / "tok.pt"
    metrics = tmp_path / "m.jsonl"
    corpus = _tiny_corpus(tmp_path)
    base = [str(PY), str(REPO / "scripts/train_vqvae.py"),
            "--config", str(REPO / "tests/fixtures/vqvae_tiny.yaml"),
            "--corpus-dir", str(corpus), "--bpe-path",
            str(tmp_path / "bpe.json"), "--out", str(out),
            "--ckpt", str(ckpt), "--metrics", str(metrics),
            "--bpe-vocab", "256", "--steps", "30", "--batch-size", "8",
            "--save-every", "15", "--eval-every", "15",
            "--threads", "2"]
    # first run: 30 steps, saves at 15 and 30
    r = subprocess.run(base, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    # second run with --steps 45 resumes from 30
    r2 = subprocess.run(base + ["--steps", "45"], capture_output=True,
                        text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert "from step 30" in r2.stdout
    rows = [json.loads(l) for l in metrics.read_text().splitlines()]
    steps = [row["step"] for row in rows if "step" in row]
    assert max(steps) >= 45
    assert (out).exists()
