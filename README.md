# MLX Speculative Code Surgeon (`mlx-code-surgeon`)

Fast, zero-swap local code infilling and formatting worker engineered for **Apple Silicon M1 (16 GB Unified Memory, 68.25 GB/s bandwidth)**.

## Hardware-Calibrated Tiers
| Tier | Target Model | Parallel Speculative Draft | Speed on M1 | Peak Metal RAM | Offline Load |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ultra` | `Qwen2.5-Coder-0.5B-Instruct-4bit` | None | **~140 tok/s** | **0.38 GB** | `0.36s` |
| `fast` *(default)* | `Qwen2.5-Coder-3B-Instruct-4bit` | `0.5B-Instruct-4bit` | **~35 tok/s** | **2.22 GB** | `0.85s` |
| `deep` | `Qwen2.5-Coder-7B-Instruct-4bit` | `0.5B-Instruct-4bit` | **~17 tok/s** | **4.77 GB** | `0.95s` |

## Usage
```bash
cd /Users/zerbytheboss/.gemini/antigravity/scratch/mlx-code-surgeon

# 1. Run Apple M1 Metal 3-Tier Benchmark
/opt/homebrew/bin/uv run python surgeon.py bench

# 2. Middle-out Fill-In-The-Middle (FIM) code infilling (--tier ultra | fast | deep)
/opt/homebrew/bin/uv run python surgeon.py infill \
  --tier fast \
  --prefix "def fibonacci(n: int) -> int:\n" \
  --suffix "\n    return a"

# 3. Infill directly from a file containing `# [INFILL]`
/opt/homebrew/bin/uv run python surgeon.py infill --tier fast --file path/to/file.py

# 4. Fast local code refactoring / formatting
/opt/homebrew/bin/uv run python surgeon.py format \
  --tier fast \
  "Convert this dict to a typed Python @dataclass: {'symbol': 'AAPL', 'price': 231.5, 'qty': 100}"
```
