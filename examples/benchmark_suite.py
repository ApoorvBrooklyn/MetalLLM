import argparse
import json
import statistics
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from metal_llm import load as load_metal


def load_baseline(model_id: str, device: str, dtype: str):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
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


def measure_once(fn, *args, tokens: int):
    t0 = time.perf_counter()
    _ = fn(*args, tokens)
    dt = time.perf_counter() - t0
    return dt, tokens / dt if dt > 0 else 0.0


def multi_trials(fn, *args, tokens: int, warmup: int, trials: int):
    for _ in range(warmup):
        measure_once(fn, *args, tokens=tokens)
    latencies, tps = [], []
    for _ in range(trials):
        dt, rate = measure_once(fn, *args, tokens=tokens)
        latencies.append(dt)
        tps.append(rate)
    return {
        "tokens": tokens,
        "latency_median_s": statistics.median(latencies),
        "latency_mean_s": statistics.fmean(latencies),
        "tps_median": statistics.median(tps),
        "tps_mean": statistics.fmean(tps),
    }


def run_suite(model_id: str, device: str, dtype: str, page_sizes, contexts, warmup: int, trials: int):
    results = {"model": model_id, "device": device, "dtype": dtype, "runs": []}

    # Baseline
    tok_b, model_b, dev_b = load_baseline(model_id, device, dtype)
    for ctx in contexts:
        prompt_short = "Hello from suite."
        prompt_long = "A" * ctx
        run = {"context": ctx, "baseline": {}}
        run["baseline"]["latency"] = [
            multi_trials(lambda p, n: generate_baseline(tok_b, model_b, dev_b, p, n), prompt_short, tokens=t, warmup=warmup, trials=trials)
            for t in (1, 8, 64)
        ]
        run["baseline"]["throughput"] = multi_trials(lambda p, n: generate_baseline(tok_b, model_b, dev_b, p, n), prompt_long, tokens=128, warmup=warmup, trials=trials)

        # MetalLLM across page sizes and streaming on/off
        run["metal_llm"] = {}
        for ps in page_sizes:
            handle = load_metal(model_id, device=device, dtype=dtype, use_paged_kv=True, page_tokens=ps)
            key = f"page_{ps}"
            run["metal_llm"][key] = {}
            # streaming on
            run["metal_llm"][key]["streaming_on"] = {
                "latency": [
                    multi_trials(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=True), prompt_short, tokens=t, warmup=warmup, trials=trials)
                    for t in (1, 8, 64)
                ],
                "throughput": multi_trials(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=True), prompt_long, tokens=128, warmup=warmup, trials=trials),
            }
            # streaming off
            run["metal_llm"][key]["streaming_off"] = {
                "latency": [
                    multi_trials(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=False), prompt_short, tokens=t, warmup=warmup, trials=trials)
                    for t in (1, 8, 64)
                ],
                "throughput": multi_trials(lambda p, n: handle.generate(p, max_new_tokens=n, use_streaming_attention=False), prompt_long, tokens=128, warmup=warmup, trials=trials),
            }
        results["runs"].append(run)

    return results


def main():
    ap = argparse.ArgumentParser(description="Benchmark suite: baseline vs MetalLLM across contexts and page sizes")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2-1.5B-Instruct"]) 
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--page-sizes", nargs="+", type=int, default=[512, 1024, 2048])
    ap.add_argument("--contexts", nargs="+", type=int, default=[1024, 4096, 16384])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default="./benchmark_results.json")
    args = ap.parse_args()

    all_results = []
    for model in args.models:
        res = run_suite(model, args.device, args.dtype, args.page_sizes, args.contexts, args.warmup, args.trials)
        all_results.append(res)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


