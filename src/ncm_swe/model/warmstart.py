"""Warm-start: transplant Qwen2.5-Coder-0.5B weights into our Block stack.

Maps 1:1 when our arch matches Qwen dims (embd 896, q 14×64, kv 2×64,
mlp 4864 SwiGLU, no qk-norm, RoPE base 1e6):
  ln_1          <- input_layernorm
  c_attn        <- [q_proj; k_proj; v_proj] (cat on out-dim)
  c_proj(attn)  <- o_proj
  ln_2          <- post_attention_layernorm
  gate/up/c_proj(mlp) <- gate_proj/up_proj/down_proj
We take the first N Qwen layers. Embeddings/heads are never transplanted
(different vocab) — always random-init.
"""

import json
from pathlib import Path

import torch


def _load_hf_state(qwen_dir: str) -> dict:
    qwen_dir = Path(qwen_dir)
    sf = sorted(qwen_dir.glob("*.safetensors"))
    if sf:
        from safetensors.torch import load_file
        state = {}
        for f in sf:
            state.update(load_file(str(f)))
        return state
    bin_f = sorted(qwen_dir.glob("*.bin"))
    state = {}
    for f in bin_f:
        state.update(torch.load(f, map_location="cpu", weights_only=False))
    return state


def hf_config(qwen_dir: str) -> dict:
    return json.loads((Path(qwen_dir) / "config.json").read_text())


def warmstart_blocks(blocks, qwen_dir: str, q_layers: list[int] | None = None):
    """Copy Qwen weights into `blocks` (our ModuleList). Returns report dict."""
    state = _load_hf_state(qwen_dir)
    cfg = hf_config(qwen_dir)
    n_q = cfg.get("num_hidden_layers", 24)
    q_layers = q_layers or list(range(len(blocks)))
    assert len(q_layers) == len(blocks), "one Qwen layer per our block"
    report = {"qwen_dir": qwen_dir, "n_qwen_layers": n_q,
              "mapped_layers": q_layers, "tensors": 0, "skipped": []}
    for dst_i, src_i in enumerate(q_layers):
        blk = blocks[dst_i]
        p = f"model.layers.{src_i}."
        def cp(dst, name):
            src = state.get(p + name)
            if src is None or tuple(src.shape) != tuple(dst.shape):
                report["skipped"].append(p + name)
                return False
            dst.data.copy_(src.to(dst.dtype))
            report["tensors"] += 1
            return True
        cp(blk.ln_1.weight, "input_layernorm.weight")
        cp(blk.ln_2.weight, "post_attention_layernorm.weight")
        q = state.get(p + "self_attn.q_proj.weight")
        k = state.get(p + "self_attn.k_proj.weight")
        v = state.get(p + "self_attn.v_proj.weight")
        if q is not None and k is not None and v is not None:
            w = torch.cat([q, k, v], dim=0)
            if tuple(w.shape) == tuple(blk.attn.c_attn.weight.shape):
                blk.attn.c_attn.weight.data.copy_(w.to(blk.attn.c_attn.weight.dtype))
                report["tensors"] += 1
            else:
                report["skipped"].append(p + "qkv(shape-mismatch)")
            if blk.attn.c_attn.bias is not None:
                bq = state.get(p + "self_attn.q_proj.bias")
                bk = state.get(p + "self_attn.k_proj.bias")
                bv = state.get(p + "self_attn.v_proj.bias")
                if bq is not None:
                    blk.attn.c_attn.bias.data.copy_(
                        torch.cat([bq, bk, bv]).to(blk.attn.c_attn.bias.dtype))
                    report["tensors"] += 1
        else:
            report["skipped"].append(p + "qkv(missing)")
        cp(blk.attn.c_proj.weight, "self_attn.o_proj.weight")
        if blk.attn.c_proj.bias is not None and \
                state.get(p + "self_attn.o_proj.bias") is not None:
            cp(blk.attn.c_proj.bias, "self_attn.o_proj.bias")
        if blk.mlp.glu:
            cp(blk.mlp.gate.weight, "mlp.gate_proj.weight")
            cp(blk.mlp.up.weight, "mlp.up_proj.weight")
        else:
            cp(blk.mlp.c_fc.weight, "mlp.gate_proj.weight")
        cp(blk.mlp.c_proj.weight, "mlp.down_proj.weight")
    return report
