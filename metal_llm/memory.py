"""
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple, List
import threading
import queue

import torch

from .storage import DiskCache
from .profiler import Profiler


@dataclass
class KVPage:
    layer_idx: int
    start_pos: int
    end_pos: int
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None


@dataclass
class PlannerOutput:
    kv_page_tokens: int
    num_hot_pages: int
    gpu_budget_bytes: int
    keep_layers_gpu: Tuple[int, int] | None  # (first_X) or (last_Y) encoded as (+X, 0) or (0, +Y)


class MemoryPlanner:
    """
    Computes paging parameters given device memory and requested context.

    Algorithm outline:
    1) Decide resident weights subset (placeholder)
    2) Compute KV page size from tokens_per_page * hidden_size * 2 * bytes
    3) Derive number of hot pages that fit within budget
    4) Adjust page size if needed to guarantee >=1 hot page
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config

    def plan(
        self,
        *,
        model_hidden_size: int,
        bytes_per_elem: int,
        requested_context: int,
        tokens_per_page: int = 1024,
        device_memory_budget_bytes: int = 3 * 1024 * 1024 * 1024,
        safety_factor: float = 1.3,
        keep_last_layers_gpu: int = 0,
        keep_first_layers_gpu: int = 0,
    ) -> PlannerOutput:
        kv_per_token = model_hidden_size * 2 * bytes_per_elem
        page_size_bytes = tokens_per_page * kv_per_token
        if page_size_bytes <= 0:
            page_size_bytes = 1

        raw_hot = int(device_memory_budget_bytes / (page_size_bytes * safety_factor))
        num_hot_pages = max(1, raw_hot)

        keep_layers_gpu: Tuple[int, int] | None = None
        if keep_first_layers_gpu > 0:
            keep_layers_gpu = (keep_first_layers_gpu, 0)
        elif keep_last_layers_gpu > 0:
            keep_layers_gpu = (0, keep_last_layers_gpu)

        return PlannerOutput(
            kv_page_tokens=tokens_per_page,
            num_hot_pages=num_hot_pages,
            gpu_budget_bytes=device_memory_budget_bytes,
            keep_layers_gpu=keep_layers_gpu,
        )


class ContextManager:
    """
    Tracks context length and is responsible for KV policy.

    For now, KV pages are RAM-only; we maintain metadata so later we can swap to
    disk-backed pages without breaking API.
    """

    def __init__(self, config: RuntimeConfig, profiler: Optional[Any] = None, disk_cache: Optional[DiskCache] = None):
        self.config = config
        self.profiler: Optional[Profiler] = profiler
        self.total_tokens = 0
        self.pages: Dict[int, list[KVPage]] = {}
        # Future: persistent KV directories, reuse across sessions
        self.persist_path: Optional[str] = None
        # Planner output cached
        self.plan_output: Optional[PlannerOutput] = None
        # Disk cache and background workers
        self.disk_cache = disk_cache
        self._bg_threads: List[threading.Thread] = []
        self._task_q: "queue.Queue[Tuple[str, Tuple[Any, ...]]]" = queue.Queue()
        self._stop_event = threading.Event()
        if self.disk_cache is not None:
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._bg_threads.append(t)

    def on_prefill(self, num_tokens: int) -> None:
        self.total_tokens += num_tokens
        # Optionally allocate/track pages based on current plan
        if self.plan_output:
            page_tokens = self.plan_output.kv_page_tokens
            current_page_id = (self.total_tokens // page_tokens) * page_tokens
            for layer_idx in self.pages.keys():
                self._ensure_page(layer_idx, current_page_id, page_tokens)

    def on_decode_step(self, new_tokens: int = 1) -> None:
        self.total_tokens += new_tokens
        # Pin current page for each layer (page by start_pos == current page start)
        if self.disk_cache and self.plan_output:
            for layer_idx, pages in self.pages.items():
                # Find page containing current tail position
                for p in pages:
                    if p.start_pos <= self.total_tokens <= p.end_pos:
                        self.disk_cache.set_pinned(layer_idx, page_id=p.start_pos, pinned=True)
                        break
        # Optionally allocate the next page if boundary crossed
        if self.plan_output:
            page_tokens = self.plan_output.kv_page_tokens
            current_page_id = (self.total_tokens // page_tokens) * page_tokens
            for layer_idx in self.pages.keys():
                self._ensure_page(layer_idx, current_page_id, page_tokens)

    def allocate_page(self, layer_idx: int, start_pos: int, end_pos: int) -> KVPage:
        page = KVPage(layer_idx=layer_idx, start_pos=start_pos, end_pos=end_pos)
        self.pages.setdefault(layer_idx, []).append(page)
        return page

    def evict_if_needed(self) -> None:
        if not self.disk_cache or not self.plan_output:
            return
        num_hot = self.plan_output.num_hot_pages
        for layer_idx, pages in self.pages.items():
            pages_sorted = sorted(pages, key=lambda p: p.start_pos)
            # Count resident (in CPU or GPU) pages we hold
            resident = [p for p in pages_sorted if (p.keys is not None or p.values is not None)]
            if len(resident) <= num_hot:
                continue
            # Queue eviction for the coldest extras beyond hot window
            cold_excess = resident[:-num_hot]
            for p in cold_excess:
                self._task_q.put(("evict", (layer_idx, p)))
        return

    # ===== Future public APIs =====
    def set_persistent_dir(self, path: str) -> None:
        """Enable KV persistence across runs."""
        self.persist_path = path
        # If disk cache is present, point it to the new path
        if self.disk_cache and path:
            try:
                os.makedirs(path, exist_ok=True)
                self.disk_cache.config.cache_dir = path
            except Exception:
                pass

    def reuse_previous_pages(self) -> None:
        """Load previous KV pages from persistent storage when applicable."""
        if not self.disk_cache or not self.plan_output:
            return
        # Prefetch last N hot pages for each known layer
        num_hot = self.plan_output.num_hot_pages
        for layer_idx, pages in self.pages.items():
            pages_sorted = sorted(pages, key=lambda p: p.start_pos)
            for p in pages_sorted[-num_hot:]:
                self._task_q.put(("prefetch", (layer_idx, p)))
        return

    # ===== Planner integration =====
    def set_plan(self, plan: PlannerOutput) -> None:
        self.plan_output = plan

    # ===== Prefetch / Evict hooks =====
    def prefetch_next_pages(self) -> None:
        if not self.disk_cache or not self.plan_output:
            return
        # Heuristic: prefetch the next hot pages for each layer
        num_hot = self.plan_output.num_hot_pages
        for layer_idx, pages in self.pages.items():
            # Identify next pages by position
            pages_sorted = sorted(pages, key=lambda p: p.start_pos)
            hot = pages_sorted[-num_hot:]
            for p in hot:
                self._task_q.put(("prefetch", (layer_idx, p)))

    def evict_cold_pages(self) -> None:
        if not self.disk_cache or not self.plan_output:
            return
        num_hot = self.plan_output.num_hot_pages
        for layer_idx, pages in self.pages.items():
            pages_sorted = sorted(pages, key=lambda p: p.start_pos)
            cold = pages_sorted[:-num_hot]
            for p in cold:
                self._task_q.put(("evict", (layer_idx, p)))

    # ===== Background worker =====
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task, args = self._task_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if task == "prefetch":
                self._do_prefetch(*args)
            elif task == "evict":
                self._do_evict(*args)
            self._task_q.task_done()

    def _do_prefetch(self, layer_idx: int, page: KVPage) -> None:
        # Load from disk into CPU (and caller may move to GPU when needed)
        try:
            tensors = self.disk_cache.load_layer(layer_idx, device="cpu")
            if tensors is not None:
                page.keys, page.values = tensors
                self.disk_cache.set_state(layer_idx, page_id=page.start_pos, state="cpu")
                if self.profiler: self.profiler.add_prefetch(1)
        except Exception:
            pass

    def _do_evict(self, layer_idx: int, page: KVPage) -> None:
        try:
            # Skip pinned pages
            if (layer_idx, page.start_pos) in (self.disk_cache.pinned_pages if self.disk_cache else set()):
                return
            # Save to disk and free CPU copies
            if page.keys is not None and page.values is not None:
                self.disk_cache.save_layer(layer_idx, page.keys, page.values)
                page.keys, page.values = None, None
                self.disk_cache.set_state(layer_idx, page_id=page.start_pos, state="disk")
                if self.profiler: self.profiler.add_evict(1)
        except Exception:
            pass

    # Cleanup
    def close(self) -> None:
        self._stop_event.set()
        for t in self._bg_threads:
            t.join(timeout=0.5)

    # ===== Helpers =====
    def _ensure_page(self, layer_idx: int, page_start: int, page_tokens: int) -> None:
        layer_pages = self.pages.setdefault(layer_idx, [])
        for p in layer_pages:
            if p.start_pos == page_start:
                return
        self.allocate_page(layer_idx, page_start, page_start + page_tokens - 1)


