import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "ncm_panel",
    Path(__file__).resolve().parents[1] / "scripts" / "ncm_panel.py",
)
npanel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(npanel)


def test_dup4_rate_no_repetition():
    texts = ["alpha beta gamma delta epsilon zeta eta theta"]
    assert npanel.dup4_rate(texts) == 0.0


def test_dup4_rate_full_repetition():
    # "a b c d" x10 -> 40 words, 37 four-grams, only 4 unique (phase shifts)
    texts = ["a b c d " * 10]
    assert npanel.dup4_rate(texts) == pytest.approx(1 - 4 / 37)


def test_dup4_rate_partial():
    # 8 words -> 5 four-grams; grams 0-2 unique, grams 3-4 duplicated
    texts = ["a b c d e f g h a b c d"]
    r = npanel.dup4_rate(texts)
    assert 0.0 < r < 1.0


def test_unicode_frac():
    assert npanel.unicode_frac("abc") == 0.0
    assert npanel.unicode_frac("açúcar") == pytest.approx(2 / 6)
    assert npanel.unicode_frac("") == 0.0


def test_distinct_ratio():
    assert npanel.distinct_ratio("a b c") == 1.0
    assert npanel.distinct_ratio("a a a") == pytest.approx(1 / 3)
    assert npanel.distinct_ratio("") == 0.0


def test_text_metrics_keys():
    m = npanel.text_metrics(["hello world foo bar", "x y z"])
    assert set(m) == {"dup4_rate", "unicode_frac", "distinct_ratio", "mean_chars"}
    assert m["mean_chars"] > 0


def test_pick_tokens_shape_and_range():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    ids = npanel.pick_tokens(logits)
    assert ids.shape == (2, 5)
    assert ids.min() >= 0 and ids.max() < 11


def test_pick_tokens_argmax_path():
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 2] = 5.0
    logits[0, 1, 1] = 5.0
    logits[0, 2, 3] = 5.0
    ids = npanel.pick_tokens(logits)
    assert ids.tolist() == [[2, 1, 3]]


def test_pick_tokens_freq_penalty_changes_pick():
    # Two identical logit rows; freq penalty should push second pick away
    logits = torch.zeros(1, 2, 3)
    logits[0, 0, 0] = 2.0
    logits[0, 1, 0] = 2.0
    logits[0, 1, 1] = 1.9
    base = npanel.pick_tokens(logits)
    penalized = npanel.pick_tokens(logits, freq=10.0)
    assert base[0, 1].item() == 0
    assert penalized[0, 1].item() == 1


def test_pick_tokens_adj_penalty():
    logits = torch.zeros(1, 3, 3)
    logits[:, :, 0] = 1.0
    logits[:, :, 1] = 0.9
    ids = npanel.pick_tokens(logits, adj=(5.0, 2, 1.0))
    # t0->0; t1 penalizes 0 -> 1; t2 penalizes both 0 and 1 -> 2
    assert ids.tolist() == [[0, 1, 2]]


def test_l0_source_shapes():
    p = torch.tensor([0.7, 0.2, 0.1])
    for mode in ("unigram", "uniform", "active", "smooth:0.1"):
        l0 = npanel.l0_source(mode, p, 2, 3, 3)
        assert l0.shape == (2, 3)
        assert l0.min() >= 0 and l0.max() < 3


def test_l0_source_model_returns_none():
    assert npanel.l0_source("model", torch.ones(4) / 4, 2, 3, 4) is None


def test_l0_source_invalid_mode():
    with pytest.raises(ValueError):
        npanel.l0_source("bogus", torch.ones(4) / 4, 1, 3, 4)


def test_parse_adj():
    assert npanel.parse_adj("") is None
    assert npanel.parse_adj("0.5,2,0.9") == [0.5, 2.0, 0.9]


def test_selftest_runs(capsys):
    npanel.selftest()
    assert "selftest OK" in capsys.readouterr().out
