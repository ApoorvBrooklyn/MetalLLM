from metal_llm import load


def main():
    handle = load("Qwen/Qwen2-1.5B-Instruct", device="mps", dtype="float16")
    prompt = "What is the Attention Mechanism in LLMs?"
    out = handle.generate(prompt, max_new_tokens=1024)
    print(out)
    print(handle.profiler.summary())


if __name__ == "__main__":
    main()


