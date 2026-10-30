import torch


def test_torch_available():
    assert torch.__version__
    assert not torch.cuda.is_available()  # CPU-only experiment


def test_vendor_tree_present():
    from ncm_swe._vendor import ensure_vendor_path

    root = ensure_vendor_path()
    assert (root / "models" / "quant.py").exists()
    assert (root / "models" / "vqvae.py").exists()
    assert (root / "LICENSE").exists()
