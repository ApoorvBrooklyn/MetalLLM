import argparse
import sys

from .api import load


def main(argv=None):
    parser = argparse.ArgumentParser(prog="metal-llm", description="MetalLLM CLI")
    parser.add_argument("prompt", type=str, nargs="?", default=None, help="Prompt text")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cpu"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--mode", type=str, default="balanced", choices=["tiny", "balanced", "high_throughput"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    if not args.prompt:
        print("Enter prompt, end with Ctrl-D (EOF):\n", file=sys.stderr)
        try:
            args.prompt = sys.stdin.read()
        except KeyboardInterrupt:
            return 1

    handle = load(args.model, device=args.device, dtype=args.dtype, mode=args.mode)
    out = handle.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


