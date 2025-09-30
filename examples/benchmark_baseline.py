import argparse
import time
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_baseline(model_id: str, device: str = "mps", dtype: str = "float16"):
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return tok, model, device


@torch.inference_mode()
def generate(tok, model, device, prompt: str, max_new_tokens: int):
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text


def bench_latency(tok, model, device, prompt: str, tokens: int) -> dict:
    t0 = time.perf_counter()
    _ = generate(tok, model, device, prompt, tokens)
    dt = time.perf_counter() - t0
    return {"tokens": tokens, "latency_s": dt, "tps": tokens / dt if dt > 0 else 0}


def bench_throughput(tok, model, device, prompt: str, tokens: int) -> dict:
    t0 = time.perf_counter()
    _ = generate(tok, model, device, prompt, tokens)
    dt = time.perf_counter() - t0
    return {"tokens": tokens, "throughput_tps": tokens / dt if dt > 0 else 0}


def main():
    ap = argparse.ArgumentParser(description="HF baseline benchmarks (no MetalLLM)")
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--context", type=int, default=1024)
    args = ap.parse_args()

    tok, model, device = load_baseline(args.model, args.device, args.dtype)

    prompt_short = "Hello from baseline."
    prompt_long = "A" * args.context

    results = {
        "latency": [
            bench_latency(tok, model, device, prompt_short, 1),
            bench_latency(tok, model, device, prompt_short, 8),
            bench_latency(tok, model, device, prompt_short, 64),
        ],
        "throughput": bench_throughput(tok, model, device, prompt_long, 128),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


