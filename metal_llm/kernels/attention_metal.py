"""
"""

from __future__ import annotations

import torch
from typing import Optional

from .attention_mps import streaming_softmax_attention


def _has_metal_attention_op() -> bool:
    """
    Check if a custom Metal kernel has been registered under torch.ops.
    Expected op schema (example): torch.ops.metal_llm.attention(q, k, v, q_block_size, k_block_size, causal:int, eps:float)
    """
    try:
        _ = torch.ops.metal_llm.attention
        return True
    except (AttributeError, RuntimeError):
        return False


def load_metal_kernels(shared_lib_path: Optional[str] = None) -> bool:
    """
    Attempt to dynamically load a compiled Metal kernel library (optional during development).
    If shared_lib_path is provided, torch.ops.load_library(shared_lib_path) will be called.
    Returns True if the attention op is available afterwards.
    """
    try:
        if shared_lib_path:
            torch.ops.load_library(shared_lib_path)
    except Exception:
        pass
    return _has_metal_attention_op()


def metal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_block_size: int = 1024,
    k_block_size: int = 1024,
    causal: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Metal-backed flash-attention-like kernel with streaming softmax.

    Fallback strategy:
    - If custom Metal kernel is unavailable, use streaming_softmax_attention (MPS-friendly).
    """
    if _has_metal_attention_op():
        causal_i = 1 if causal else 0
        return torch.ops.metal_llm.attention(q, k, v, q_block_size, k_block_size, causal_i, float(eps))
    # Fallback to streaming MPS implementation
    return streaming_softmax_attention(
        q, k, v, q_block_size=q_block_size, k_block_size=k_block_size, causal=causal, eps=eps
    )


