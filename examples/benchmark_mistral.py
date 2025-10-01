from metal_llm import load
import time, json


def main():
    handle = load("mistralai/Mistral-7B-v0.1", device="mps", dtype="float16", use_paged_kv=True, install_attn_hooks=True)
    prompt = "Benchmarking Mistral paged KV."
    t0 = time.perf_counter()
    out = handle.generate(prompt, max_new_tokens=256, use_streaming_attention=True)
    dt = time.perf_counter() - t0
    handle.save_profile("./bench_mistral_trace.json")
    print(json.dumps({"model": "mistral", "tok": 256, "latency_s": dt, "tps": 256 / dt if dt > 0 else 0}, indent=2))


if __name__ == "__main__":
    main()


