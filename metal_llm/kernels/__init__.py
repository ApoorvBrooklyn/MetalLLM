"""
Kernel Library overview:
- MPS path: reference implementations using PyTorch ops (attention, etc.)
- Metal path: future high-performance MSL kernels (flash-attn, fused MLP, layernorm)

Registry helpers will allow runtime to choose kernels per op.
"""

from .attention_mps import mps_attention, streaming_softmax_attention
from .attention_metal import metal_attention, load_metal_kernels

REGISTRY = {
    "attention": {
        "mps": mps_attention,
        "metal": metal_attention,
        "mps_streaming": streaming_softmax_attention,
    }
}

__all__ = [
    "mps_attention",
    "streaming_softmax_attention",
    "metal_attention",
    "load_metal_kernels",
    "REGISTRY",
]


