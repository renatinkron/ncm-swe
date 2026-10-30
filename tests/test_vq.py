import pytest
import torch

from ncm_swe.vq.model import build_vqvae, load_vqvae, save_vqvae

TINY = {
    "in_vocab_size": 100,
    "context_length": 16,
    "embed_dim": 32,
    "vqvae_vocab_size": [64, 64, 64],
    "n_layers": 2,
    "compression_factor": 1,
    "dropout": 0.0,
    "pre_quant_groupnorm": 4,
    "enc_layers": 1,
    "dec_layers": 1,
    "use_qk_norm": True,
    "enc_positional": True,
    "dec_positional": True,
    "vector_quantizer": {
        "clss": "multiscale_residual_vector_quantizer",
        "decay": 0.99,
        "commitment_cost": 0.25,
        "restart_threshold": 0.0,
        "levels": {"use_manual_levels": True, "manual_levels": [1, 4, 16]},
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
        "aux": {},
    },
}


def test_vqvae_shapes_and_levels():
    model = build_vqvae(TINY)
    x = torch.randint(0, 100, (2, 16))
    logits, vq_loss, per_lvl, glob_util, _ = model(x)
    assert logits.shape == (2, 16, 100)
    assert torch.isfinite(vq_loss)
    assert len(per_lvl) == 3 and all(0 <= u <= 1 for u in per_lvl)


def test_encode_decode_roundtrip_shapes():
    model = build_vqvae(TINY)
    model.eval()
    x = torch.randint(0, 100, (3, 16))
    idx = model.encode(x)
    # sum(levels) = 1 + 4 + 16 = 21 codes
    assert idx.shape == (3, 21)
    assert idx.max() < 64 and idx.min() >= 0
    z_q = model.vq.idx_to_z_q(idx)
    assert z_q.shape == (3, 16, 32)
    out = model.decode(idx)
    assert out.shape == (3, 16, 100)


def test_ncm_inputs_contract():
    model = build_vqvae(TINY)
    model.eval()
    x = torch.randint(0, 100, (2, 16))
    ncm_in, targets = model.get_ncm_inputs(x)
    # inputs: chunks of lengths levels[1:] = 4+16 = 20 positions, dim=embed_dim
    assert ncm_in.shape == (2, 20, 32)
    # targets: codes for levels 1.. -> idx[:, levels[0]:] = 4+16 = 20 codes
    assert targets.shape == (2, 20)


def test_save_load(tmp_path):
    model = build_vqvae(TINY)
    model.eval()
    x = torch.randint(0, 100, (2, 16))
    idx0 = model.encode(x)
    save_vqvae(model, tmp_path / "t.pt", TINY)
    m2, _, _ = load_vqvae(tmp_path / "t.pt")
    m2.eval()
    assert torch.equal(idx0, m2.encode(x))
