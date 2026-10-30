"""Path shim for the pristine vendored upstream NCM code.

Vendored files live in ``vendor/ncm-upstream`` and import via repo-root-style
paths (``models.*``, ``utils.*``, ``tokenization.*``). We never edit them in
place; we expose the upstream root on ``sys.path`` once here and re-export the
symbols our code needs.
"""

import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "ncm-upstream"


def ensure_vendor_path() -> Path:
    p = str(_VENDOR_ROOT)
    if _VENDOR_ROOT.is_dir() and p not in sys.path:
        sys.path.insert(0, p)
    return _VENDOR_ROOT


def upstream():
    """Return the key upstream classes (import lazily so tests can skip)."""
    ensure_vendor_path()
    from models.vqvae import VQVAE
    from models.quant import MultiscaleResidualVectorQuantizer
    from models.next_concept import NextConceptTransformer

    return VQVAE, MultiscaleResidualVectorQuantizer, NextConceptTransformer
