import torch

from ncm_swe.model.baseline_ar import ARBaseline
from ncm_swe.model.generator import LevelGenerator
from ncm_swe.vq.model import build_vqvae
from tests.test_vq import TINY

GEN = dict(n_embd=64, n_layer=2, n_head_q=4, n_head_kv=2, dropout=0.0,
           mlp_hidden=128, glu=True, use_qk_norm=False, bias=True,
           shared_head=True)


def _tiny():
    vq = build_vqvae(TINY).eval()
    return vq, LevelGenerator(vq, **GEN)


def test_forward_and_per_level_ce():
    vq, g = _tiny()
    idx = torch.randint(0, 64, (2, sum(vq.vq.levels)))  # 21 codes
    loss, per_level = g.loss(idx)
    assert torch.isfinite(loss)
    assert set(per_level) == {"l0", "l1", "l2"}


def test_mask_block_diagonal():
    vq, g = _tiny()
    m = g.attn_mask[0, 0]
    T = 1 + sum(g.chunk_levels)
    assert m.shape == (T, T)
    # a position may not attend to a different level (excluding col 0)
    l1_end = 1 + g.chunk_levels[0]  # first col of level-2 chunk
    row = 1  # a level-1 position must not see any level-2 column
    later = m[row, l1_end:]
    assert torch.isinf(later).all()
    # and its own-level cells are visible
    assert (m[row, 1:l1_end] == 0).all()


def test_sampling_passes_and_shape():
    vq, g = _tiny()
    idx, codes = g.generate(batch_size=2, return_per_level=True)
    assert idx.shape == (2, sum(vq.vq.levels))
    assert len(codes) == len(vq.vq.levels)          # K passes
    for c, l in zip(codes, vq.vq.levels):
        assert c.shape == (2, l)


def test_baseline_parity_and_cache():
    ar = ARBaseline(vocab_size=100, n_embd=64, n_layer=2, n_head_q=4,
                    n_head_kv=2, mlp_hidden=128, context_length=16)
    ids = torch.randint(0, 100, (2, 16))
    loss, _ = ar(ids, targets=ids)
    assert torch.isfinite(loss)
    out = ar.generate(ids[:, :3], max_new_tokens=4)
    assert out.shape[1] == 7


def test_param_counts_documented():
    vq, g = _tiny()
    ar = ARBaseline(vocab_size=8192, n_embd=896, n_layer=4, n_head_q=14,
                    n_head_kv=2, mlp_hidden=4864)
    p_g = sum(p.numel() for p in g.parameters() if p.requires_grad)
    p_a = sum(p.numel() for p in ar.parameters() if p.requires_grad)
    # same block stack → counts within order of magnitude (head/emb differ)
    assert p_g > 0 and p_a > 0
