"""
Custom attention processor for HuggingFace Qwen2 that routes attention to
MetalLLM kernels and integrates paging signals.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..memory import ContextManager
from ..profiler import Profiler
from ..kernels import REGISTRY


class MetalLLMQwenAttnProcessor(nn.Module):
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
        bsz, q_len, _ = hidden_states.size()
        num_heads = getattr(attn, "num_heads", None)
        num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
        head_dim = getattr(attn, "head_dim", None)

        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        def shape(x, n_heads):
            return x.view(bsz, -1, n_heads, head_dim).transpose(1, 2)

        q = shape(q, num_heads)
        k = shape(k, num_kv_heads)
        v = shape(v, num_kv_heads)

        rotary = getattr(attn, "rotary_emb", None)
        if rotary is not None:
            try:
                q, k = rotary(q, k, position_ids)
            except TypeError:
                q, k = rotary(q, k)

        if num_kv_heads != num_heads:
            repeat = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        use_streaming = bool(self.context_manager.config.use_paged_kv)
        kernel_key = "mps_streaming" if use_streaming else "mps"
        attn_fn = REGISTRY["attention"].get(kernel_key) or next(iter(REGISTRY["attention"].values()))

        if self.profiler is not None:
            t0 = self.profiler.now()

        if use_streaming and self.context_manager.plan_output is not None:
            layer_idx = getattr(attn, "layer_idx", 0)
            current_page = self.context_manager.current_page_id()
            self.context_manager.register_layer(layer_idx)
            self.context_manager.pin_page(layer_idx, current_page, device=q.device.type)

            kv = self.context_manager.assemble_kv_window(layer_idx, max_tokens=self.context_manager.config.prefill_block_size, device=q.device.type)
            if kv is not None:
                past_k, past_v = kv
                k = torch.cat([past_k.to(k.dtype), k], dim=-2)
                v = torch.cat([past_v.to(v.dtype), v], dim=-2)

        context = attn_fn(q, k, v, causal=True)
        if self.profiler is not None:
            self.profiler.mark("qwen_attention", t0)

        context = context.transpose(1, 2).contiguous().view(bsz, q_len, num_heads * head_dim)
        out = attn.o_proj(context)

        if use_streaming and q_len == 1:
            layer_idx = getattr(attn, "layer_idx", 0)
            self.context_manager.append_kv(layer_idx, k[..., -1:, :], v[..., -1:, :])

        if output_attentions:
            return out, None, None if use_cache else out
        if use_cache:
            return out, None
        return out


def install_qwen_attention_processor(model: nn.Module, context_manager: ContextManager, profiler: Optional[Profiler] = None) -> int:
    replaced = 0
    proc = MetalLLMQwenAttnProcessor(context_manager=context_manager, profiler=profiler)
    for module in model.modules():
        if hasattr(module, "attn_processor") and hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj") and hasattr(module, "o_proj"):
            try:
                module.attn_processor = proc
                replaced += 1
            except Exception:
                continue
    return replaced


