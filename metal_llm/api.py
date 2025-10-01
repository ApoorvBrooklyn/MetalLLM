"""
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any, Generator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from .runtime import RuntimeConfig, select_device, select_dtype, apply_mode_defaults, make_memory_planner
from .memory import ContextManager, PlannerOutput
from .storage import DiskCache, DiskCacheConfig
from .profiler import Profiler
from .attn_hooks import install_model_attention_hooks, SUPPORTED_MODEL_TYPES
from .hf_processors.llama import install_llama_attention_processor
from .hf_processors.qwen import install_qwen_attention_processor
from .hf_processors.mistral import install_mistral_attention_processor


@dataclass
class ModelHandle:
    model: Any
    tokenizer: Any
    config: RuntimeConfig
    context_manager: ContextManager
    profiler: Profiler
    tokenizer_cache: Dict[str, Any] = field(default_factory=dict)

    def profile_json(self) -> str:
        return self.profiler.to_chrome_trace()

    def save_profile(self, path: str) -> None:
        self.profiler.save_chrome_trace(path)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = True,
        callback: Optional[Callable[[str], None]] = None,
        seed: Optional[int] = None,
        context_manager: Optional[ContextManager] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        use_streaming_attention: Optional[bool] = None,
        **gen_kwargs: Any,
    ) -> str:
        """
        Minimal streaming generation using HF generate + TextIteratorStreamer.

        Notes:
        - Works on MPS and CPU. If MPS is not available, falls back to CPU.
        - ContextManager is currently a stub; future work will integrate paged KV.
        - For large context, prefer smaller batch sizes and careful max length.
        """
        # Optional override of context manager
        if context_manager is not None:
            self.context_manager = context_manager
        # Optional switch for streaming attention path (routes via scheduler)
        if use_streaming_attention is not None:
            self.config.use_paged_kv = bool(use_streaming_attention)

        device = self.config.device
        tokenizer = self.tokenizer
        model = self.model

        # Tokenizer caching (simple in-memory by prompt string)
        t0 = self.profiler.now()
        cache_key = prompt
        inputs = self.tokenizer_cache.get(cache_key)
        if inputs is None:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            )
            self.tokenizer_cache[cache_key] = inputs
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.profiler.mark("tokenize", t0)

        # Context hooks (stubbed)
        self.context_manager.on_prefill(num_tokens=inputs["input_ids"].shape[-1])

        streamer = None
        if stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            streamer=streamer,
            **gen_kwargs,
        )

        if seed is not None:
            torch.manual_seed(seed)
            try:
                import random; random.seed(seed)
                import numpy as np; np.random.seed(seed % (2**32 - 1))
            except Exception:
                pass
        self.model.eval()
        with torch.inference_mode():
            if streamer is None:
                t0 = self.profiler.now()
                outputs = model.generate(**inputs, **generate_kwargs)
                self.profiler.mark("generate_total", t0)
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if on_complete:
                    try:
                        on_complete(self.profiler.summary())
                    except Exception:
                        pass
                return text
            else:
                # Launch generation in a background thread (HF handles it internally)
                t0 = self.profiler.now()
                _ = model.generate(**inputs, **generate_kwargs)
                acc = []
                token_idx = 0
                for token_text in streamer:
                    acc.append(token_text)
                    if callback:
                        callback(token_text)
                    self.profiler.add_tokens(1)
                    self.context_manager.on_decode_step(new_tokens=1)
                    token_idx += 1
                    # periodic prefetch/evict & pin current page
                    if token_idx % 8 == 0:
                        self.context_manager.prefetch_next_pages()
                        self.context_manager.evict_cold_pages()
                self.profiler.mark("generate_total", t0)
                out = "".join(acc)
                if on_complete:
                    try:
                        on_complete(self.profiler.summary())
                    except Exception:
                        pass
                return out


def load(
    model_id: str,
    *,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    mode: str = "balanced",
    trust_remote_code: bool = True,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    # Storage / paging knobs
    disk_cache_dir: str = "./kv_cache",
    page_tokens: int = 1024,
    device_memory_budget_bytes: int = 3 * 1024 * 1024 * 1024,
    use_paged_kv: bool = False,
    install_attn_hooks: bool = False,
) -> ModelHandle:
    """
    Load a HuggingFace causal LM on macOS using MPS when available.

    Parameters
    - model_id: HF model id or local directory path.
    - device: "mps" | "cpu". Defaults to MPS if available.
    - dtype: "float16" | "bfloat16" | "float32". Chooses best for MPS.
    - mode: "tiny" | "balanced" | "high_throughput" (affects planner settings).

    Returns a ModelHandle with .generate().
    """
    dev = select_device(device)
    dt = select_dtype(dtype, device=dev)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dt]

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
        low_cpu_mem_usage=True,
    )

    model.to(dev)

    cfg = apply_mode_defaults(RuntimeConfig(device=dev, dtype=dt, mode=mode))
    cfg.use_paged_kv = bool(use_paged_kv)
    profiler = Profiler(enable=True)
    # Disk-backed KV cache integration stub
    disk_cache = DiskCache(DiskCacheConfig(cache_dir=disk_cache_dir))
    ctx_mgr = ContextManager(config=cfg, profiler=profiler, disk_cache=disk_cache)
    # Derive initial memory plan (placeholder numbers; integrate real queries later)
    hidden_size_guess = getattr(getattr(model, "config", None), "hidden_size", None) or getattr(model.config, "d_model", 4096)
    bytes_per_elem = 2 if dt == "float16" else 4
    planner = make_memory_planner(cfg)
    plan: PlannerOutput = planner.plan(
        model_hidden_size=hidden_size_guess,
        bytes_per_elem=bytes_per_elem,
        requested_context=cfg.max_context,
        tokens_per_page=page_tokens,
        device_memory_budget_bytes=device_memory_budget_bytes,
        safety_factor=1.3,
        keep_last_layers_gpu=0,
        keep_first_layers_gpu=0,
    )
    ctx_mgr.set_plan(plan)

    handle = ModelHandle(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        context_manager=ctx_mgr,
        profiler=profiler,
    )

    # Optionally install attention hooks for supported model types
    if install_attn_hooks:
        try:
            model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
            if model_type in SUPPORTED_MODEL_TYPES:
                install_model_attention_hooks(model, model_type=model_type, context_manager=ctx_mgr, profiler=profiler)
                if model_type == "llama":
                    install_llama_attention_processor(model, ctx_mgr, profiler)
                elif model_type == "qwen2":
                    install_qwen_attention_processor(model, ctx_mgr, profiler)
                elif model_type == "mistral":
                    install_mistral_attention_processor(model, ctx_mgr, profiler)
        except Exception:
            pass

    return handle


def generate(*args, **kwargs) -> str:
    """
    Convenience wrapper: load-then-generate single-shot.
    """
    handle = load(*args, **kwargs)
    return handle.generate(kwargs.get("prompt", ""))


# ===== Convenience helpers on ModelHandle =====
def set_kv_persistence(handle: ModelHandle, path: str) -> None:
    handle.context_manager.set_persistent_dir(path)


def recompute_memory_plan(
    handle: ModelHandle,
    *,
    page_tokens: Optional[int] = None,
    device_memory_budget_bytes: Optional[int] = None,
) -> None:
    hidden_size = getattr(getattr(handle.model, "config", None), "hidden_size", None) or getattr(handle.model.config, "d_model", 4096)
    bytes_per_elem = 2 if handle.config.dtype == "float16" else 4
    planner = make_memory_planner(handle.config)
    plan: PlannerOutput = planner.plan(
        model_hidden_size=hidden_size,
        bytes_per_elem=bytes_per_elem,
        requested_context=handle.config.max_context,
        tokens_per_page=page_tokens or 1024,
        device_memory_budget_bytes=device_memory_budget_bytes or (3 * 1024 * 1024 * 1024),
        safety_factor=1.3,
    )
    handle.context_manager.set_plan(plan)


