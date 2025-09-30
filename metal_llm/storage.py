"""
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch


@dataclass
class DiskCacheConfig:
    cache_dir: str = "./kv_cache"
    readonly: bool = False


class DiskCache:
    """
    Disk-backed KV cache placeholder. Today it only prepares directories and a
    simple save/load pair. Later will manage page files and indexes.
    """

    def __init__(self, config: DiskCacheConfig):
        self.config = config
        os.makedirs(self.config.cache_dir, exist_ok=True)
        # Track page states and pins
        self.pinned_pages: set[tuple[int, int]] = set()  # (layer_idx, page_id)
        self.page_state: dict[tuple[int, int], str] = {}  # 'gpu' | 'cpu' | 'disk'

    def _path(self, layer_idx: int) -> str:
        return os.path.join(self.config.cache_dir, f"layer_{layer_idx}.pt")

    def _path_page(self, layer_idx: int, page_id: int) -> str:
        return os.path.join(self.config.cache_dir, f"layer_{layer_idx}_page_{page_id}.pt")

    def save_layer(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if self.config.readonly:
            return
        t1 = time.perf_counter()
        torch.save((keys.cpu(), values.cpu()), self._path(layer_idx))
        _ = time.perf_counter() - t1

    def load_layer(self, layer_idx: int, device: str = "cpu") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        path = self._path(layer_idx)
        if not os.path.exists(path):
            return None
        t1 = time.perf_counter()
        keys, values = torch.load(path, map_location=device)
        _ = time.perf_counter() - t1
        return keys, values

    # Per-page save/load for PagedKV
    def save_page(self, layer_idx: int, page_id: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if self.config.readonly:
            return
        path = self._path_page(layer_idx, page_id)
        t1 = time.perf_counter()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save((keys.cpu(), values.cpu()), path)
        _ = time.perf_counter() - t1

    def load_page(self, layer_idx: int, page_id: int, device: str = "cpu") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        path = self._path_page(layer_idx, page_id)
        if not os.path.exists(path):
            return None
        t1 = time.perf_counter()
        keys, values = torch.load(path, map_location=device)
        _ = time.perf_counter() - t1
        return keys, values

    # Page state management
    def set_pinned(self, layer_idx: int, page_id: int, pinned: bool = True) -> None:
        key = (layer_idx, page_id)
        if pinned:
            self.pinned_pages.add(key)
        else:
            self.pinned_pages.discard(key)

    def set_state(self, layer_idx: int, page_id: int, state: str) -> None:
        self.page_state[(layer_idx, page_id)] = state



# ===== Future: Safetensors / checkpoint streaming loader =====
class SafeTensorsStreamLoader:
    """
    Stream safetensors without OS mmap to avoid page cache pressure.
    Design:
      - Open files and read necessary slices into preallocated CPU buffers
      - Pin and move to MPS/CPU as needed
      - Provide iterator or random-access API for tensors
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.index: Dict[str, str] = {}
        self._files: Dict[str, "_SafeTensorReader"] = {}
        index_json = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_json):
            import json
            with open(index_json) as f:
                j = json.load(f)
                # weight_map: tensor_name -> filename
                self.index = j.get("weight_map", {})
        else:
            # Fallback: single safetensors file named model.safetensors
            self.index = {}

    def _reader(self, filename: str):
        if filename not in self._files:
            path = os.path.join(self.model_dir, filename)
            self._files[filename] = _SafeTensorReader(path)
        return self._files[filename]

    def get_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        # Resolve which file contains this tensor
        filename = self.index.get(name, "model.safetensors")
        reader = self._reader(filename)
        t = reader.get_tensor(name)
        return t.to(device)

    def close(self):
        for r in self._files.values():
            r.close()
        self._files.clear()


class _SafeTensorReader:
    """Minimal safetensors file reader without OS mmap."""
    def __init__(self, path: str):
        import struct, json
        self.path = path
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = f.read(header_len)
            self.header = json.loads(header)
            self.data_offset = 8 + header_len
        self._fp = open(path, "rb")
        self._dtype_map: Dict[str, Any] = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
            "I8": torch.int8,
            "I32": torch.int32,
        }

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self.header[name]
        dtype = self._dtype_map[info["dtype"]]
        shape = info["shape"]
        off0, off1 = info["data_offsets"]
        self._fp.seek(self.data_offset + off0)
        buf = self._fp.read(off1 - off0)
        t = torch.frombuffer(memoryview(buf), dtype=dtype).reshape(shape)
        return t


class LRUPageCache:
    """
    LRU cache plan for KV pages on disk.
    - Track hot pages in memory, evict oldest to disk when over budget
    - Prefetch predicted next pages for smoother decoding
    """

    def __init__(self, max_in_memory_pages: int = 8):
        self.max_in_memory_pages = max_in_memory_pages
        self.cache: Dict[Any, Any] = {}
        self.order: list[Any] = []
        self.on_evict: Optional[Callable[[Any, Any], None]] = None  # callback(key, value)

    def touch(self, key):
        if key in self.order:
            self.order.remove(key)
        self.order.insert(0, key)

    def evict_if_needed(self):
        while len(self.order) > self.max_in_memory_pages:
            victim = self.order.pop()
            val = self.cache.pop(victim, None)
            if self.on_evict and val is not None:
                try:
                    self.on_evict(victim, val)
                except Exception:
                    pass


