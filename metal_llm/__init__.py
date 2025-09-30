"""
MetalLLM: macOS-first large-context LLM inference library.

This package targets Apple Silicon machines using PyTorch MPS and custom Metal
kernels (future work) to enable long-context inference without quantization.

Public API surface is intentionally small and HuggingFace-like. Prefer importing
high-level functions and classes from `metal_llm.api`.
"""

from .api import load, generate, ContextManager

__all__ = [
    "load",
    "generate",
    "ContextManager",
]

__version__ = "0.0.1"


