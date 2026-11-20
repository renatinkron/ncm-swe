import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dump_knowledge",
    ROOT / "scripts" / "dump_knowledge.py",
)
dk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dk)

DUMP = ROOT / "data" / "lora_dump" / "knowledge_dump.jsonl"


def test_real_dump_validates():
    errors, stats = dk.validate(str(DUMP))
    assert errors == []
    assert stats["total"] >= dk.MIN_ENTRIES
    assert stats["by_lang"]["en"] > 0 and stats["by_lang"]["pt"] > 0


def test_missing_file_errors(tmp_path):
    errors, _ = dk.validate(str(tmp_path / "nope.jsonl"))
    assert errors


def test_bad_json_flagged(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id":"x","lang":"en","domain":"logic","topic":"t","content":"c"}\nnot json\n')
    errors, _ = dk.validate(str(p))
    assert any("invalid JSON" in e for e in errors)


def test_missing_field_flagged(tmp_path):
    p = tmp_path / "miss.jsonl"
    p.write_text(json.dumps({"id": "x", "lang": "en"}) + "\n")
    errors, _ = dk.validate(str(p))
    assert any("missing fields" in e for e in errors)


def test_pt_in_programming_domain_flagged(tmp_path):
    row = {"id": "x1", "lang": "pt", "domain": "algorithms",
           "topic": "t", "content": "conteúdo de algoritmo em português"}
    p = tmp_path / "ptprog.jsonl"
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    errors, _ = dk.validate(str(p))
    assert any("unexpected 'pt'" in e for e in errors)


def test_duplicate_id_flagged(tmp_path):
    row = {"id": "dup", "lang": "en", "domain": "logic",
           "topic": "t", "content": "c"}
    p = tmp_path / "dup.jsonl"
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    errors, _ = dk.validate(str(p))
    assert any("duplicate id" in e for e in errors)
