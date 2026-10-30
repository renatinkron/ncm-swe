import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_corpus",
    Path(__file__).resolve().parents[1] / "scripts" / "build_corpus.py",
)
bc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bc)


def _records(tmp_path):
    return [
        {"source": "seed", "meta": {"i": 0}, "text": "alpha text one"},
        {"source": "seed", "meta": {"i": 1}, "text": "bravo text two"},
        {"source": "seed", "meta": {"i": 2}, "text": "alpha text one"},  # dup
    ]


def test_shard_schema_and_dedup(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = bc.load_manifest(corpus)
    n = bc.write_records(corpus, manifest, _records(tmp_path), shard_size=10)
    assert n == 2  # dedup removed the repeated text
    lines = (corpus / manifest["shards"][0]).read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert set(rec) >= {"id", "source", "text", "meta"}
        assert rec["id"].startswith("seed-")


def test_resume_skips_completed_stage(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = bc.load_manifest(corpus)
    bc.write_records(corpus, manifest, _records(tmp_path), shard_size=10)
    bc.mark_stage(manifest, "seed", 2, 24)
    (corpus / "manifest.json").write_text(json.dumps(manifest))
    manifest2 = bc.load_manifest(corpus)
    assert bc.stage_done(manifest2, "seed")
    # second write of same records appends nothing (dedup across shards)
    n = bc.write_records(corpus, manifest2, _records(tmp_path), shard_size=10)
    assert n == 0


def test_template_records_nonempty():
    recs = bc.gen_template_records(2)
    assert len(recs) == 2 * len(bc.TASK_BANK)
    assert all(r["source"] == "template" and len(r["text"]) > 10 for r in recs)
