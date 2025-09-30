import argparse
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from metal_llm import load as load_metal


def load_baseline(model_id: str, device: str, dtype: str):
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
def generate_baseline(tok, model, device, prompt: str, max_new_tokens: int):
    inputs = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


def measure_latency(fn, *args, tokens: int):
    t0 = time.perf_counter()
    _ = fn(*args, tokens)
    dt = time.perf_counter() - t0
    return {"tokens": tokens, "latency_s": dt, "tps": tokens / dt if dt > 0 else 0}


def main():
    ap = argparse.ArgumentParser(description="Compare MetalLLM vs baseline (Transformers)")
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--page", type=int, default=1024)
    ap.add_argument("--profile", default="./compare_trace.json")
    args = ap.parse_args()

    # Baseline
    tok_b, model_b, dev_b = load_baseline(args.model, args.device, args.dtype)
    prompt_short = "Hello from benchmark."
    prompt_long = "A" * args.context

    base_lat = [
        measure_latency(lambda p, n: generate_baseline(tok_b, model_b, dev_b, p, n), prompt_short, tokens=1),
        measure_latency(lambda p, n: generate_baseline(tok_b, model_b, dev_b, p, n), prompt_short, tokens=8),
        measure_latency(lambda p, n: generate_baseline(tok_b, model_b, dev_b, p, n), prompt_short, tokens=64),
    ]
    t0 = time.perf_counter(); _ = generate_baseline(tok_b, model_b, dev_b, prompt_long, 128); base_tp = 128 / (time.perf_counter() - t0)

    # MetalLLM
    handle = load_metal(
        args.model,
        device=args.device,
        dtype=args.dtype,
        use_paged_kv=True,
        page_tokens=args.page,
    )
    metal_lat = [
        measure_latency(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=True), prompt_short, tokens=1),
        measure_latency(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=True), prompt_short, tokens=8),
        measure_latency(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=True), prompt_short, tokens=64),
    ]
    t0 = time.perf_counter(); _ = handle.generate(prompt_long, max_new_tokens=128, use_streaming_attention=True); metal_tp = 128 / (time.perf_counter() - t0)
    handle.save_profile(args.profile)

    result = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "baseline": {"latency": base_lat, "throughput_tps": base_tp},
        "metal_llm": {"latency": metal_lat, "throughput_tps": metal_tp, "profiler_summary": handle.profiler.summary()},
        "trace": args.profile,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


