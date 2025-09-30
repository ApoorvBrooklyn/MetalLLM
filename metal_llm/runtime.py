"""
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable

import torch
import time

from .memory import MemoryPlanner
from . import kernels

@dataclass
class RuntimeConfig:
    device: str = "mps"  # "mps" or "cpu"
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32"
    mode: str = "balanced"  # "tiny" | "balanced" | "high_throughput"
    max_context: int = 100_000
    # Tunables (subject to change as planner evolves)
    prefill_block_size: int = 2048
    decode_batch_size: int = 1
    use_paged_kv: bool = False
    enable_streaming: bool = True


def select_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(dtype: Optional[str] = None, device: Optional[str] = None) -> str:
    if dtype:
        return dtype
    dev = device or select_device(None)
    # On MPS, float16 performs well. bfloat16 is improving but still evolving.
    if dev == "mps":
        return "float16"
    return "float32"


class Scheduler:
    """
    Placeholder for an execution planner which will:
    - Monitor VRAM (unified memory usage) and CPU RAM
    - Decide KV placement: GPU (MPS), CPU, or Disk pages
    - Configure attention kernel paths
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._last_plan_ctx = 0
        self._last_memory_check = 0.0
        self._cached_memory: Dict[str, Any] = {}

    def plan_step(self, current_ctx: int, *, context_manager=None, profiler=None) -> None:
        """
        Periodic runtime planning hook. Decides whether to enable streaming/paged
        attention and triggers prefetch/evict based on hot window.
        """
        # Update memory stats occasionally
        now = time.time()
        if now - self._last_memory_check > 0.25:
            self._cached_memory = self._query_memory()
            self._last_memory_check = now

        # Enable paged KV beyond a threshold of context length
        if current_ctx > (self.config.prefill_block_size * 2):
            self.config.use_paged_kv = True

        # Trigger prefetch/evict if a context window advanced sufficiently
        if context_manager is not None and current_ctx - self._last_plan_ctx >= 32:
            context_manager.prefetch_next_pages()
            context_manager.evict_cold_pages()
            self._last_plan_ctx = current_ctx
            if profiler:
                # Mark a small span to reflect planning step
                t0 = profiler.now(); profiler.mark("planner_step", t0)

    def select_kernel(self, op: str) -> str:
        """
        Return which kernel path to use for a given op.
        Returns 'mps' by default. For attention, can return 'mps_streaming' when
        long contexts are in use or configured via mode.
        """
        if op == "attention" and self.config.use_paged_kv:
            return "mps_streaming"
        return "mps"

    def resolve_kernel(self, op: str) -> Callable[..., Any]:
        """
        Return the callable for the selected kernel from the registry.
        """
        key = self.select_kernel(op)
        table = kernels.REGISTRY.get(op, {})
        fn = table.get(key)
        if fn is None:
            # fallback to any available implementation
            return next(iter(table.values()))
        return fn

    def _query_memory(self) -> Dict[str, Any]:
        """
        Query current memory stats (best-effort). On macOS, PyTorch MPS exposes
        current allocated memory; unified memory makes exact figures approximate.
        """
        stats: Dict[str, Any] = {}
        try:
            if self.config.device == "mps" and hasattr(torch, "mps"):
                stats["mps_allocated"] = int(torch.mps.current_allocated_memory())
                stats["mps_reserved"] = int(torch.mps.driver_allocated_memory()) if hasattr(torch.mps, "driver_allocated_memory") else None
        except Exception:
            pass
        return stats


def apply_mode_defaults(cfg: RuntimeConfig) -> RuntimeConfig:
    """
    Populate config tunables based on selected mode.
    - tiny: prioritize low memory, smaller blocks, may reduce throughput
    - balanced: default trade-offs
    - high_throughput: larger prefill blocks, streaming enabled
    """
    mode = (cfg.mode or "balanced").lower()
    if mode == "tiny":
        cfg.prefill_block_size = 1024
        cfg.decode_batch_size = 1
        cfg.use_paged_kv = False
        cfg.enable_streaming = True
    elif mode == "high_throughput":
        cfg.prefill_block_size = 4096
        cfg.decode_batch_size = 1
        cfg.use_paged_kv = False
        cfg.enable_streaming = True
    else:  # balanced
        cfg.prefill_block_size = 2048
        cfg.decode_batch_size = 1
        cfg.use_paged_kv = False
        cfg.enable_streaming = True
    return cfg


def make_memory_planner(cfg: RuntimeConfig) -> MemoryPlanner:
    return MemoryPlanner(cfg)


