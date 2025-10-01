import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from metal_llm import load


def run_hf(model_id: str, prompt: str, max_new_tokens: int = 128):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mod = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    mod.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mod.to(device)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    t0 = time.perf_counter()
    out_ids = mod.generate(**inputs, max_new_tokens=max_new_tokens)
    dt = time.perf_counter() - t0
    return tok.decode(out_ids[0], skip_special_tokens=True), dt


def run_metal(model_id: str, prompt: str, max_new_tokens: int = 128):
    handle = load(model_id, device="mps", dtype="float16", use_paged_kv=True, install_attn_hooks=True)
    t0 = time.perf_counter()
    text = handle.generate(prompt, max_new_tokens=max_new_tokens, use_streaming_attention=True)
    dt = time.perf_counter() - t0
    return text, dt


def main():
    cases = [
        ("meta-llama/Llama-2-7b-chat-hf", "Parity test for Llama."),
        ("Qwen/Qwen2-1.5B-Instruct", "Parity test for Qwen."),
        ("mistralai/Mistral-7B-v0.1", "Parity test for Mistral."),
    ]
    results = []
    for model_id, prompt in cases:
        ref, dt_ref = run_hf(model_id, prompt)
        test, dt_test = run_metal(model_id, prompt)
        # Very loose textual parity metric: Jaccard of token sets
        ref_set = set(ref.split())
        test_set = set(test.split())
        jacc = len(ref_set & test_set) / max(1, len(ref_set | test_set))
        results.append({
            "model": model_id,
            "jaccard": round(jacc, 3),
            "latency_ref_s": dt_ref,
            "latency_test_s": dt_test,
            "tps_test": 128 / dt_test if dt_test > 0 else 0,
        })
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()


