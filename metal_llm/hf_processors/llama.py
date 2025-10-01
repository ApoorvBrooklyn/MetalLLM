"""
Custom attention processor for HuggingFace Llama that routes attention to
MetalLLM kernels and integrates paging signals.

This is an initial implementation targeting HF Llama v2/v3 attention modules
that expose projections (q_proj, k_proj, v_proj, o_proj) and rotary embeddings.
It falls back to a safe MPS path if Metal is unavailable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..memory import ContextManager
from ..profiler import Profiler
from ..kernels import REGISTRY


class MetalLLMLlamaAttnProcessor(nn.Module):
    def __init__(self, context_manager: ContextManager, profiler: Optional[Profiler] = None):
        super().__init__()
        self.context_manager = context_manager
        self.profiler = profiler

    @torch.no_grad()
    def forward(
        self,
        attn: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # Shapes and projections
        bsz, q_len, _ = hidden_states.size()
        num_heads = getattr(attn, "num_heads", None)
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = getattr(attn, "head_dim", None)

        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        # Reshape to (B, num_heads, T, D)
        def shape(x, n_heads):
            return x.view(bsz, -1, n_heads, head_dim).transpose(1, 2)

        q = shape(q, num_heads)
        k = shape(k, num_kv_heads)
        v = shape(v, num_kv_heads)

        # Apply RoPE if present
        rotary = getattr(attn, "rotary_emb", None)
        if rotary is not None:
            # Many HF Llama modules expose rotary_emb as a callable
            try:
                q, k = rotary(q, k, position_ids)
            except TypeError:
                # Fallback: some variants use (q, k) only
                q, k = rotary(q, k)

        # Expand KV heads to match Q heads when using grouped-query attention
        if num_kv_heads != num_heads:
            repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Runtime selection of kernel
        use_streaming = bool(self.context_manager.config.use_paged_kv)
        kernel_key = "mps_streaming" if use_streaming else "mps"
        attn_fn = REGISTRY["attention"].get(kernel_key)
        if attn_fn is None:
            # Fallback to any available impl
            attn_fn = next(iter(REGISTRY["attention"].values()))

        if self.profiler is not None:
            t0 = self.profiler.now()

        # If paged KV enabled, attempt to fetch hot KV pages (prefetch already triggered by hooks)
        if use_streaming and self.context_manager.plan_output is not None:
            # Pin current page for this layer
            layer_idx = getattr(attn, "layer_idx", 0)
            current_page = self.context_manager.current_page_id()
            self.context_manager.register_layer(layer_idx)
            self.context_manager.pin_page(layer_idx, current_page, device=q.device.type)

            # Assemble KV window to use as past for long contexts
            kv = self.context_manager.assemble_kv_window(layer_idx, max_tokens=self.context_manager.config.prefill_block_size, device=q.device.type)
            if kv is not None:
                past_k, past_v = kv
                # Concatenate along time for attention
                k = torch.cat([past_k.to(k.dtype), k], dim=-2)
                v = torch.cat([past_v.to(v.dtype), v], dim=-2)

        # Call kernel: returns (B, H, Tq, D)
        context = attn_fn(q, k, v, causal=True)

        if self.profiler is not None:
            self.profiler.mark("llama_attention", t0)

        # Merge heads back and project out
        context = context.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        out = attn.o_proj(context)

        if use_streaming:
            # Append KV for the newly generated token(s) into current page
            # For prefill (q_len>1) we skip append in this minimal path.
            if q_len == 1:
                layer_idx = getattr(attn, "layer_idx", 0)
                self.context_manager.append_kv(layer_idx, k[..., -1:, :], v[..., -1:, :])

        if output_attentions:
            # Not supported in this path; return None for attn_weights
            return out, None, None if use_cache else out
        if use_cache:
            # Return KV cache placeholder for API compatibility
            return out, None
        return out


def install_llama_attention_processor(model: nn.Module, context_manager: ContextManager, profiler: Optional[Profiler] = None) -> int:
    """Replace attention processors on Llama attention modules when supported.

    Returns number of modules updated.
    """
    replaced = 0
    proc = MetalLLMLlamaAttnProcessor(context_manager=context_manager, profiler=profiler)
    for module in model.modules():
        # Newer HF exposes .attn_processor field; ensure API compatibility
        if hasattr(module, "attn_processor") and hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj") and hasattr(module, "o_proj"):
            try:
                module.attn_processor = proc
                replaced += 1
            except Exception:
                continue
    return replaced


