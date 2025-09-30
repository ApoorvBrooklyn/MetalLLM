import argparse
import time
import json

from metal_llm import load


def bench_latency(handle, prompt: str, tokens: int) -> dict:
    t0 = time.perf_counter()
    _ = handle.generate(prompt, max_new_tokens=tokens, use_streaming_attention=True)
    dt = time.perf_counter() - t0
    return {"tokens": tokens, "latency_s": dt, "tps": tokens / dt if dt > 0 else 0}


def bench_throughput(handle, prompt: str, tokens: int) -> dict:
    t0 = time.perf_counter()
    _ = handle.generate(prompt, max_new_tokens=tokens, use_streaming_attention=True)
    dt = time.perf_counter() - t0
    return {"tokens": tokens, "throughput_tps": tokens / dt if dt > 0 else 0}


def bench_memory_paging(handle) -> dict:
    # Inspect profiler counters summary
    summary = handle.profiler.summary()
    return {"profiler_summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--context", default=1024, type=int)
    ap.add_argument("--page", default=1024, type=int)
    ap.add_argument("--profile", default="./bench_trace.json")
    args = ap.parse_args()

    handle = load(
        args.model,
        device=args.device,
        dtype=args.dtype,
        mode=args.mode,
        use_paged_kv=True,
        page_tokens=args.page,
    )

    prompt_short = "Hello from MetalLLM."
    prompt_long = "A" * args.context

    results = {
        "latency": [
            bench_latency(handle, prompt_short, 1),
            bench_latency(handle, prompt_short, 8),
            bench_latency(handle, prompt_short, 64),
        ],
        "throughput": bench_throughput(handle, prompt_long, 128),
        "memory_paging": bench_memory_paging(handle),
    }

    handle.save_profile(args.profile)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


