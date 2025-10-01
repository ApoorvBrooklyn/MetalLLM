"""
Attention hook scaffolding for HF models (Llama, Qwen, Mistral).

Current goals:
- Do not change math yet. Install lightweight hooks to integrate paging signals
  and prepare for kernel-backed overrides.
- Detect attention modules and register pre/post hooks to drive ContextManager
  prefetch/evict and add profiler spans.

Future work:
- Replace attention kernel calls inside target attention modules to use
  Metal/MPS streaming kernels with KV paging.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Callable

import torch

from .profiler import Profiler
from .memory import ContextManager


SUPPORTED_MODEL_TYPES: Tuple[str, ...] = (
    "llama",
    "qwen2",
    "mistral",
)


def _is_attention_module(name: str, module: torch.nn.Module, model_type: str) -> bool:
    n = name.lower()
    mt = (model_type or "").lower()
    if mt == "llama":
        return n.endswith(".self_attn") or n.endswith(".attention") or n.endswith(".attn")
    if mt == "qwen2":
        return n.endswith(".attention") or n.endswith(".self_attn") or "qwen2" in n and "attn" in n
    if mt == "mistral":
        return n.endswith(".self_attn") or n.endswith(".mha") or "attention" in n
    # Fallback heuristic
    return hasattr(module, "num_heads") and hasattr(module, "forward")


def install_model_attention_hooks(
    model: torch.nn.Module,
    *,
    model_type: str,
    context_manager: ContextManager,
    profiler: Optional[Profiler] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Install lightweight forward hooks on attention modules to integrate paging signals.

    Returns a list of hook handles. Caller should keep them for later removal if needed.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _pre_hook(mod: torch.nn.Module, inputs):
        # inputs typically: (hidden_states, attention_mask, position_ids, ...)
        # We only use token count to drive paging heuristics.
        try:
            context_manager.prefetch_next_pages()
            context_manager.evict_cold_pages()
            if profiler is not None:
                t0 = profiler.now(); profiler.mark("attn_pre", t0)
        except Exception:
            pass

    def _post_hook(mod: torch.nn.Module, inputs, output):
        try:
            if profiler is not None:
                t0 = profiler.now(); profiler.mark("attn_post", t0)
        except Exception:
            pass
        return output

    # Assign stable layer indices to attention modules in discovery order
    layer_counter = 0
    for name, module in model.named_modules():
        if _is_attention_module(name, module, model_type):
            try:
                setattr(module, "layer_idx", layer_counter)
                context_manager.register_layer(layer_counter)
                layer_counter += 1
            except Exception:
                pass
            try:
                handles.append(module.register_forward_pre_hook(_pre_hook, with_kwargs=False))
                handles.append(module.register_forward_hook(_post_hook, with_kwargs=False))
            except Exception:
                continue

    return handles



