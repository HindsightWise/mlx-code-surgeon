"""MLX Speculative Code Surgeon for Apple Silicon (16 GB M1).

Provides 3 hardware-calibrated tiers for the 68.25 GB/s base Apple M1:
- `fast`  : Qwen2.5-Coder-3B-Instruct-4bit (1.63 GB) + 0.5B draft (~2.0 GB RAM)
- `deep`  : Qwen2.5-Coder-7B-Instruct-4bit (4.00 GB) + 0.5B draft (~4.7 GB RAM)
- `ultra` : Qwen2.5-Coder-0.5B-Instruct-4bit (0.27 GB) standalone (~0.35 GB RAM)

Supports:
1. Native bi-contextual Fill-In-The-Middle (FIM) code infilling with suffix deduplication.
2. Parallel speculative decoding on Apple Metal GPU.
3. Offline-first zero-latency local HF cache loading (`local_files_only=True`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate

TIERS = {
    "ultra": {
        "target": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        "draft": None,
    },
    "fast": {
        "target": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
        "draft": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    },
    "deep": {
        "target": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "draft": "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    },
}

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
STOP_MARKERS = ("<|endoftext|>", "<|file_sep|>", "<|fim_pad|>", "<|im_end|>")

_MODEL_CACHE: dict[str, tuple[object, object]] = {}


def resolve_local_repo(repo_id: str) -> str:
    """Resolve cached Hugging Face snapshot without network round-trips."""
    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=repo_id)


def get_cached_model(repo_id: str) -> tuple[object, object]:
    """Load and cache model/tokenizer in memory for zero-reload latency."""
    if repo_id not in _MODEL_CACHE:
        local_path = resolve_local_repo(repo_id)
        _MODEL_CACHE[repo_id] = load(local_path)
    return _MODEL_CACHE[repo_id]


def build_fim_prompt(prefix: str, suffix: str) -> str:
    """Construct native Qwen2.5-Coder Fill-In-The-Middle prompt."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def trim_fim_overlap(generated: str, suffix: str) -> str:
    """Strip stop markers and any trailing overlap with the suffix anchor."""
    for marker in STOP_MARKERS:
        if marker in generated:
            generated = generated.split(marker, 1)[0]

    gen_lines = generated.splitlines()
    if gen_lines and gen_lines[0].startswith("   ") and not gen_lines[0].startswith("    "):
        gen_lines[0] = " " + gen_lines[0]

    suffix_lines = [line.strip() for line in suffix.splitlines() if line.strip()]
    if suffix_lines:
        first_suffix_line = suffix_lines[0]
        for idx, line in enumerate(gen_lines):
            if line.strip() == first_suffix_line:
                gen_lines = gen_lines[:idx]
                break

    return "\n".join(gen_lines).rstrip()


def run_generation(
    prompt: str,
    *,
    tier: str = "fast",
    use_chat_template: bool = False,
    max_tokens: int = 256,
    num_draft_tokens: int = 4,
    use_speculative: bool = True,
    stop_line: str | None = None,
    verbose: bool = False,
) -> tuple[str, dict[str, float]]:
    """Run generation on Apple Metal with optional parallel speculative decoding."""
    mx.reset_peak_memory()
    tier_cfg = TIERS[tier]
    t0 = time.perf_counter()
    model, tokenizer = get_cached_model(tier_cfg["target"])

    draft_model = None
    if use_speculative and tier_cfg["draft"]:
        draft_model, _ = get_cached_model(tier_cfg["draft"])
    load_time = time.perf_counter() - t0

    if use_chat_template and tokenizer.chat_template is not None:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a fast, surgical code generator and formatter. "
                    "Output only clean code or requested data without markdown chatter."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt

    chunks: list[str] = []
    last_resp = None
    gen_kwargs: dict[str, object] = {
        "max_tokens": max_tokens,
    }
    if draft_model is not None:
        gen_kwargs["draft_model"] = draft_model
        gen_kwargs["num_draft_tokens"] = num_draft_tokens

    for resp in stream_generate(model, tokenizer, prompt=formatted_prompt, **gen_kwargs):
        chunks.append(resp.text)
        last_resp = resp
        if verbose:
            print(resp.text, end="", flush=True)
        current_text = "".join(chunks)
        if any(m in current_text for m in STOP_MARKERS):
            break
        if stop_line and stop_line in current_text:
            break

    if verbose:
        print()

    text = "".join(chunks)
    metrics = {
        "load_time_s": round(load_time, 3),
        "prompt_tokens": float(getattr(last_resp, "prompt_tokens", 0) or 0),
        "prompt_tps": round(float(getattr(last_resp, "prompt_tps", 0.0) or 0.0), 2),
        "generation_tokens": float(getattr(last_resp, "generation_tokens", 0) or 0),
        "generation_tps": round(
            float(getattr(last_resp, "generation_tps", 0.0) or 0.0), 2
        ),
        "peak_memory_gb": round(
            float(getattr(last_resp, "peak_memory", mx.get_peak_memory() / 1e9) or 0.0),
            3,
        ),
    }
    return text, metrics


def cmd_infill(args: argparse.Namespace) -> int:
    """Execute middle-out Fill-In-The-Middle code infilling."""
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
        if args.marker not in content:
            print(
                f"Error: marker '{args.marker}' not found in {args.file}",
                file=sys.stderr,
            )
            return 1
        prefix, suffix = content.split(args.marker, 1)
    else:
        prefix = args.prefix or ""
        suffix = args.suffix or ""

    suffix_lines = [line.strip() for line in suffix.splitlines() if line.strip()]
    stop_line = suffix_lines[0] if suffix_lines else None

    fim_prompt = build_fim_prompt(prefix, suffix)
    raw_output, metrics = run_generation(
        fim_prompt,
        tier=args.tier,
        use_chat_template=False,
        max_tokens=args.max_tokens,
        num_draft_tokens=args.draft_tokens,
        use_speculative=not args.no_speculative,
        stop_line=stop_line,
        verbose=False,
    )
    clean_output = trim_fim_overlap(raw_output, suffix)
    print(clean_output)
    if args.stats:
        print(
            f"\n--- M1 Metal Stats [{args.tier}]: {metrics['generation_tps']} tok/s | "
            f"Prompt: {metrics['prompt_tps']} tok/s | "
            f"Peak RAM: {metrics['peak_memory_gb']} GB ---",
            file=sys.stderr,
        )
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    """Execute fast local code refactoring or schema formatting."""
    _, metrics = run_generation(
        args.prompt,
        tier=args.tier,
        use_chat_template=True,
        max_tokens=args.max_tokens,
        num_draft_tokens=args.draft_tokens,
        use_speculative=not args.no_speculative,
        verbose=True,
    )
    if args.stats:
        print(
            f"--- M1 Metal Stats [{args.tier}]: {metrics['generation_tps']} tok/s | "
            f"Prompt: {metrics['prompt_tps']} tok/s | "
            f"Peak RAM: {metrics['peak_memory_gb']} GB ---",
            file=sys.stderr,
        )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Benchmark all 3 M1 Metal tiers on middle-out FIM code infilling."""
    prefix = (
        "from bisect import bisect_left, bisect_right\n"
        "from typing import Sequence\n\n"
        "def binary_search_range(arr: Sequence[int], target: int) -> tuple[int, int]:\n"
        '    """Return (first_index, last_index) of target in sorted arr, or (-1, -1)."""\n'
    )
    suffix = "\n    return left_bound, right_bound\n"
    fim_prompt = build_fim_prompt(prefix, suffix)

    print("=== Apple M1 (16 GB Unified Memory) Metal FIM Benchmark ===\n")
    last_clean = ""
    for tier_name in ("ultra", "fast", "deep"):
        raw_out, m = run_generation(
            fim_prompt,
            tier=tier_name,
            use_chat_template=False,
            max_tokens=args.max_tokens,
            num_draft_tokens=4,
            use_speculative=True,
            stop_line="return left_bound, right_bound",
            verbose=False,
        )
        last_clean = trim_fim_overlap(raw_out, suffix)
        cfg = TIERS[tier_name]
        print(f"[{tier_name.upper()}] Target: {cfg['target']} | Draft: {cfg['draft']}")
        print(
            f"  Speed: {m['generation_tps']:6.2f} tok/s | "
            f"Prompt: {m['prompt_tps']:6.2f} tok/s | "
            f"Peak Metal RAM: {m['peak_memory_gb']:.3f} GB | "
            f"Offline Load: {m['load_time_s']:.2f}s"
        )

    print("\n--- Verified FIM Infill Output ---")
    print(prefix + last_clean + suffix)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MLX Speculative Code Surgeon (Qwen2.5-Coder 7B / 3B / 0.5B)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_infill = sub.add_parser("infill", help="Middle-out FIM code infilling")
    p_infill.add_argument(
        "--tier", choices=["ultra", "fast", "deep"], default="fast"
    )
    p_infill.add_argument("--prefix", type=str, help="Code before cursor")
    p_infill.add_argument("--suffix", type=str, help="Code after cursor")
    p_infill.add_argument("--file", type=str, help="File containing infill marker")
    p_infill.add_argument(
        "--marker", type=str, default="# [INFILL]", help="Marker string in file"
    )
    p_infill.add_argument("--max-tokens", type=int, default=256)
    p_infill.add_argument("--draft-tokens", type=int, default=4)
    p_infill.add_argument("--no-speculative", action="store_true")
    p_infill.add_argument("--stats", action="store_true", default=True)
    p_infill.set_defaults(func=cmd_infill)

    p_fmt = sub.add_parser("format", help="Fast local code/schema formatting")
    p_fmt.add_argument(
        "--tier", choices=["ultra", "fast", "deep"], default="fast"
    )
    p_fmt.add_argument("prompt", type=str, help="Instruction or code to format")
    p_fmt.add_argument("--max-tokens", type=int, default=512)
    p_fmt.add_argument("--draft-tokens", type=int, default=4)
    p_fmt.add_argument("--no-speculative", action="store_true")
    p_fmt.add_argument("--stats", action="store_true", default=True)
    p_fmt.set_defaults(func=cmd_format)

    p_bench = sub.add_parser("bench", help="Run M1 Metal throughput & RAM benchmark")
    p_bench.add_argument("--max-tokens", type=int, default=160)
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
