import argparse
import json
import time

from .api import load


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metal-llm-bench", description="MetalLLM microbenchmarks")
    ap.add_argument("--model", default="Qwen/Qwen2-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--page", type=int, default=1024)
    ap.add_argument("--profile", default="./bench_trace.json")
    args = ap.parse_args(argv)

    handle = load(
        args.model,
        device=args.device,
        dtype=args.dtype,
        use_paged_kv=True,
        page_tokens=args.page,
    )

    prompt = "Hello from MetalLLM."
    t0 = time.perf_counter()
    _ = handle.generate(prompt, max_new_tokens=64, use_streaming_attention=True)
    dt = time.perf_counter() - t0

    handle.save_profile(args.profile)
    print(json.dumps({"throughput_tps": 64 / dt if dt > 0 else 0, "latency_s": dt}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


