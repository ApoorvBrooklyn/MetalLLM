"""
Kernel Library overview:
- MPS path: reference implementations using PyTorch ops (attention, etc.)
- Metal path: future high-performance MSL kernels (flash-attn, fused MLP, layernorm)

Registry helpers will allow runtime to choose kernels per op.
"""

from .attention_mps import mps_attention, streaming_softmax_attention

REGISTRY = {
    "attention": {
        "mps": mps_attention,
        # "metal": metal_attention  # to be added when implemented
        "mps_streaming": streaming_softmax_attention,
    }
}

__all__ = ["mps_attention", "streaming_softmax_attention", "REGISTRY"]


