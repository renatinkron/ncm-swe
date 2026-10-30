"""Factory + persistence for the NCM stage-1 VQ-VAE (vendored upstream arch)."""

from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from ncm_swe._vendor import ensure_vendor_path


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_vqvae(cfg: dict):
    ensure_vendor_path()
    from models.vqvae import VQVAE

    vq_cfg = OmegaConf.create(dict(cfg["vector_quantizer"]))
    return VQVAE(
        in_vocab_size=int(cfg["in_vocab_size"]),
        context_length=int(cfg["context_length"]),
        embed_dim=int(cfg["embed_dim"]),
        vqvae_vocab_size=list(cfg["vqvae_vocab_size"]),
        n_layers=int(cfg["n_layers"]),
        vector_quantizer_config=vq_cfg,
        compression_factor=int(cfg.get("compression_factor", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        pre_quant_groupnorm=cfg.get("pre_quant_groupnorm"),
        pre_quant_dropout=cfg.get("pre_quant_dropout"),
        use_qk_norm=bool(cfg.get("use_qk_norm", False)),
        enc_layers=int(cfg.get("enc_layers", 0)),
        enc_attn_window=int(cfg.get("enc_attn_window", 0)),
        enc_conv_layers=int(cfg.get("enc_conv_layers", 0)),
        enc_conv_kernel=int(cfg.get("enc_conv_kernel", 3)),
        dec_layers=int(cfg.get("dec_layers", 1)),
        enc_positional=bool(cfg.get("enc_positional", True)),
        dec_positional=bool(cfg.get("dec_positional", False)),
    )


def save_vqvae(model, path: str | Path, cfg: dict, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "config": cfg, "meta": meta or {}},
               tmp)
    tmp.rename(path)  # atomic checkpoint


def load_vqvae(path: str | Path):
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model = build_vqvae(ckpt["config"])
    model.load_state_dict(ckpt["model"], strict=True)
    return model, ckpt["config"], ckpt.get("meta", {})
