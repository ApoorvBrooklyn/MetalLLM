"""
Simple profiling utilities: throughput, latency, memory snapshots.

Future work:
- Export traces compatible with Instruments / Chrome tracing.
- Per-layer timings and page IO stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import os
import resource
import json


@dataclass
class Profiler:
    enable: bool = True
    events: Dict[str, List[float]] = field(default_factory=dict)
    counters: Dict[str, float] = field(default_factory=dict)
    trace_events: List[Dict[str, Any]] = field(default_factory=list)  # Chrome trace format

    def mark(self, name: str, t0: float) -> None:
        if not self.enable:
            return
        self.events.setdefault(name, []).append(round(time.perf_counter() - t0, 6))
        # Add a simple complete event to trace as well
        self._add_trace_complete(name=name, dur_s=time.perf_counter() - t0, cat="phase")

    def now(self) -> float:
        return time.perf_counter()

    # ===== Context manager helpers =====
    def span(self, name: str, cat: str = "phase"):
        """
        Context manager to capture a timed span, recorded into events and chrome trace.
        Usage: with profiler.span("generate"): ...
        """
        profiler = self
        class _Span:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                self_inner.ts_us = time.time() * 1e6
                return self_inner
            def __exit__(self_inner, exc_type, exc, tb):
                if not profiler.enable:
                    return False
                dt = time.perf_counter() - self_inner.t0
                profiler.events.setdefault(name, []).append(round(dt, 6))
                profiler._add_trace_complete(name=name, dur_s=dt, cat=cat, ts_us=self_inner.ts_us)
                return False
        return _Span()

    def summary(self) -> str:
        if not self.events:
            return ""
        parts = []
        for k, v in self.events.items():
            parts.append(f"{k}: n={len(v)} sum={sum(v):.3f}s avg={(sum(v)/len(v)):.4f}s")
        mem = self._memory_summary()
        if mem:
            parts.append(mem)
        # Derived metrics
        if "generate_total" in self.events and self.counters.get("tokens", 0) > 0:
            total = sum(self.events["generate_total"]) or 1e-9
            tps = self.counters["tokens"] / total
            parts.append(f"throughput={tps:.2f} tok/s")
        if self.counters:
            parts.append("counters=" + ",".join(f"{k}:{int(v)}" for k, v in self.counters.items()))
        return " | ".join(parts)

    def _memory_summary(self) -> str:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is KB on macOS
            rss_mb = usage.ru_maxrss / 1024.0
            return f"max_rss={rss_mb:.1f}MB"
        except Exception:
            return ""

    # Public helpers
    def add_tokens(self, n: int = 1) -> None:
        self.counters["tokens"] = self.counters.get("tokens", 0) + n
    def add_spill(self, n: int = 1) -> None:
        self.counters["spill"] = self.counters.get("spill", 0) + n
    def add_prefetch(self, n: int = 1) -> None:
        self.counters["prefetch"] = self.counters.get("prefetch", 0) + n
    def add_evict(self, n: int = 1) -> None:
        self.counters["evict"] = self.counters.get("evict", 0) + n
    def add_decompress(self, n: int = 1) -> None:
        self.counters["decompress"] = self.counters.get("decompress", 0) + n

    # ===== Per-layer timings and page IO trace helpers =====
    def layer_begin(self, layer_idx: int, kind: str = "forward") -> float:
        """Mark the start of a layer operation and return start time."""
        t0 = time.perf_counter()
        self._add_trace_instant(name=f"layer_{layer_idx}_{kind}_begin", cat="layer")
        return t0

    def layer_end(self, layer_idx: int, t0: float, kind: str = "forward") -> None:
        self.mark(f"layer_{layer_idx}_{kind}", t0)

    def page_io(self, action: str, layer_idx: int, page_id: int, size_bytes: Optional[int] = None) -> None:
        """Record a page IO event: prefetch/evict/decompress/spill."""
        if action == "prefetch": self.add_prefetch(1)
        elif action == "evict": self.add_evict(1)
        elif action == "decompress": self.add_decompress(1)
        elif action == "spill": self.add_spill(1)
        self._add_trace_instant(name=f"page_{action}_L{layer_idx}_P{page_id}", cat="page", args={"bytes": size_bytes or 0})

    # ===== Chrome trace export =====
    def _add_trace_complete(self, name: str, dur_s: float, cat: str = "phase", ts_us: Optional[float] = None) -> None:
        if not self.enable:
            return
        if ts_us is None:
            ts_us = time.time() * 1e6
        self.trace_events.append({"name": name, "cat": cat, "ph": "X", "ts": ts_us, "dur": dur_s * 1e6, "pid": 0, "tid": 0})

    def _add_trace_instant(self, name: str, cat: str = "phase", args: Optional[Dict[str, Any]] = None) -> None:
        if not self.enable:
            return
        self.trace_events.append({"name": name, "cat": cat, "ph": "i", "s": "t", "ts": time.time() * 1e6, "pid": 0, "tid": 0, "args": args or {}})

    def to_chrome_trace(self) -> str:
        """Return a Chrome trace JSON string."""
        return json.dumps({"traceEvents": self.trace_events}, indent=2)

    def save_chrome_trace(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_chrome_trace())

    def reset(self) -> None:
        self.events.clear()
        self.counters.clear()
        self.trace_events.clear()


