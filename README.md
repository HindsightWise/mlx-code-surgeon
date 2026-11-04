# MLX Code Surgeon (`mlx-code-surgeon`)

**Human–AI Collaborative Engineering Build**  
- **Hardware Memory Budget & Tiered Routing Architecture**: Robert Zerby ([@Zerbytheboss](https://x.com/Zerbytheboss) | [Hindsight Wise](https://github.com/HindsightWise))  
- **Apple Metal MLX Speculative Decoding & MCP Implementation**: AI Collaborator (Claude / Gemini)

---

## Overview

In a multi-tier human–AI workflow, calling frontier cloud models for localized Fill-In-The-Middle (FIM) code infilling or deterministic schema/code formatting wastes both latency and cloud token budgets. Conversely, loading oversized local weights on a 16 GB Apple Silicon machine causes macOS memory swap collapse.

`mlx-code-surgeon` is a **System-1.5 Local Apple Metal MCP Server & CLI** built on `mlx-lm` that executes zero-swap Fill-In-The-Middle (`<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`) code infilling and batch formatting using 4-bit quantized `Qwen2.5-Coder` models paired with parallel **speculative decoding** (`0.5B` draft model accelerating `3B` and `7B` target models) and an explicit Metal RAM unload gate.

## Verified Apple M1 (16 GB Unified Memory) Benchmarks

| Tier | Target Model | Draft Model | Generation Speed | Peak Metal RAM | Load Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ultra`** | `Qwen2.5-Coder-0.5B-Instruct-4bit` | None | **93.89 tok/s** | **0.35 GB** | `0.62s` |
| **`fast`** *(Default)* | `Qwen2.5-Coder-3B-Instruct-4bit` | `0.5B-Instruct-4bit` | **29.74 tok/s** | **2.28 GB** | `0.73s` |
| **`deep`** | `Qwen2.5-Coder-7B-Instruct-4bit` | `0.5B-Instruct-4bit` | **16.58 tok/s** | **4.97 GB** | `0.95s` |

*Memory Guardrail*: Even at the `deep` 7B tier with the 0.5B speculative draft model co-resident, peak wired Metal RAM stays strictly under `5.0 GB`, preventing macOS unified memory swap pressure. Calling `mlx_surgeon_status(unload=True)` immediately clears GPU cache before heavy builds or renders.

---

## MCP Tools Exposed (`mcp_server.py`)

- `mlx_surgeon_status`: Reports active Metal memory usage (`active_metal_gb`, `peak_metal_gb`), cached models, and supports `unload=True` to free unified memory immediately.
- `mlx_surgeon_infill`: Executes surgical Fill-In-The-Middle (FIM) code completion around a `# [INFILL]` marker in an existing file or snippet.
- `mlx_surgeon_format`: Runs fast local code/schema refactoring and formatting instructions using `Qwen2.5-Coder` with speculative decoding.

## Quick Start

```bash
# 1. Run Apple Metal hardware benchmark across all 3 tiers
python3 surgeon.py bench --tier all

# 2. Surgical Fill-In-The-Middle (FIM) on a file containing '# [INFILL]'
python3 surgeon.py infill --file path/to/file.py --tier fast

# 3. Direct prefix/suffix FIM
python3 surgeon.py infill --prefix "def fib(n: int) -> int:\n    " --suffix "\n    return a" --tier ultra
```
