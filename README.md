MetalLLM (macOS-first LLM Inference)
====================================

MetalLLM is an open-source Python library that brings large-context LLM inference
to Apple Silicon Macs (M1–M4) using PyTorch MPS and future custom Metal kernels.

Goals
-----
- Run Llama, GPT-OSS, Qwen models with up to 100k context on macOS
- No quantization required (fp16/bf16/float32)
- Memory-aware planner to split KV cache across GPU/CPU/Disk
- Stream-safe attention kernels (Metal) for long contexts
- HuggingFace-like API and CoreML/Swift export for apps

Quickstart
----------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece
```

```python
from metal_llm import load

handle = load("meta-llama/Llama-2-7b-chat-hf", device="mps", dtype="float16")
out = handle.generate("Hello, summarize Metal for GPUs in 3 bullets.", max_new_tokens=64)
print(out)
```

Status
------
- MVP works on MPS with a minimal streaming generate path
- KV paging and Metal kernels in progress. 

Future Updates planned
-------
- Paged KV cache with disk offload (100k+ tokens)
- Flash-attention-like kernels in Metal (MSL)
- Memory-aware execution planner and modes (tiny/balanced/high_throughput)
- CoreML exporter + Swift package

License
-------
Apache-2.0 (TBD)


