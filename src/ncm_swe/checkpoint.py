"""Atomic, resumable checkpointing shared by all trainers."""

import time
from pathlib import Path

import torch


def save_ckpt(path: str | Path, model, opt, step: int, cfg: dict,
              extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {"model": model.state_dict(),
         "opt": opt.state_dict() if opt is not None else None,
         "step": step, "cfg": cfg, "extra": extra or {},
         "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        tmp)
    tmp.rename(path)


def load_ckpt(path: str | Path, model, opt=None) -> dict:
    ckpt = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if opt is not None and ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    return ckpt
