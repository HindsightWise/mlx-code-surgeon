"""Stdio Model Context Protocol (MCP) server for Apple Silicon M1 MLX Speculative Code Surgeon."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
from surgeon import (
    TIERS,
    _MODEL_CACHE,
    build_fim_prompt,
    resolve_local_repo,
    run_generation,
    trim_fim_overlap,
)

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "mlx_surgeon_status",
        "description": (
            "Return status, cached models, and Apple Metal memory usage for the local M1 "
            "MLX Speculative Code Surgeon (0.5B ultra @ 140 tok/s, 3B+0.5B fast @ 43 tok/s, "
            "7B+0.5B deep @ 17 tok/s)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "unload": {
                    "type": "boolean",
                    "description": "If true, clear in-memory Metal model cache to free unified RAM.",
                    "default": False,
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mlx_surgeon_infill",
        "description": (
            "Execute local Apple Metal bi-contextual Fill-In-The-Middle (FIM) code infilling "
            "using Qwen2.5-Coder + parallel speculative decoding. Provide either (prefix, suffix) "
            "or a file_path containing a marker (default '# [INFILL]'). Optionally write the "
            "infilled result directly back to the file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Code preceding the cursor/infill point.",
                },
                "suffix": {
                    "type": "string",
                    "description": "Code following the cursor/infill point.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional file path containing the infill marker.",
                },
                "marker": {
                    "type": "string",
                    "description": "Marker string to replace when file_path is used (default '# [INFILL]').",
                    "default": "# [INFILL]",
                },
                "apply_in_place": {
                    "type": "boolean",
                    "description": "If true and file_path is provided, replace marker in file_path with the generated code.",
                    "default": False,
                },
                "tier": {
                    "type": "string",
                    "enum": ["ultra", "fast", "deep"],
                    "description": "Hardware tier: 'ultra' (0.5B, 140 tok/s), 'fast' (3B+0.5B draft, 43 tok/s), 'deep' (7B+0.5B draft, 17 tok/s).",
                    "default": "fast",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate (default 256).",
                    "default": 256,
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "mlx_surgeon_format",
        "description": (
            "Execute fast local Apple Metal code refactoring, boilerplate generation, or "
            "JSON/dataclass schema formatting using Qwen2.5-Coder + parallel speculative decoding."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The refactoring, formatting, or code transform instruction.",
                },
                "tier": {
                    "type": "string",
                    "enum": ["ultra", "fast", "deep"],
                    "description": "Hardware tier: 'ultra' (0.5B, 140 tok/s), 'fast' (3B+0.5B draft, 43 tok/s), 'deep' (7B+0.5B draft, 17 tok/s).",
                    "default": "fast",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate (default 512).",
                    "default": 512,
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    },
]


def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool call for mlx-code-surgeon."""
    if name == "mlx_surgeon_status":
        if arguments.get("unload"):
            _MODEL_CACHE.clear()
            gc.collect()
            mx.clear_cache()
        cached_Snapshots = {}
        for tier_name, cfg in TIERS.items():
            cached_Snapshots[tier_name] = {
                "target": cfg["target"],
                "draft": cfg["draft"],
                "local_path": resolve_local_repo(cfg["target"]),
                "loaded_in_ram": cfg["target"] in _MODEL_CACHE,
            }
        return {
            "status": "ONLINE",
            "device": str(mx.default_device()),
            "active_memory_gb": round(mx.get_active_memory() / 1e9, 3),
            "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 3),
            "tiers": cached_Snapshots,
        }

    if name == "mlx_surgeon_infill":
        tier = arguments.get("tier", "fast")
        max_tokens = int(arguments.get("max_tokens", 256))
        marker = arguments.get("marker", "# [INFILL]")
        file_path = arguments.get("file_path")

        if file_path:
            p = Path(file_path)
            content = p.read_text(encoding="utf-8")
            if marker not in content:
                raise ValueError(f"Marker '{marker}' not found in {file_path}")
            prefix, suffix = content.split(marker, 1)
        else:
            prefix = arguments.get("prefix", "")
            suffix = arguments.get("suffix", "")

        suffix_lines = [line.strip() for line in suffix.splitlines() if line.strip()]
        stop_line = suffix_lines[0] if suffix_lines else None
        fim_prompt = build_fim_prompt(prefix, suffix)

        raw_out, metrics = run_generation(
            fim_prompt,
            tier=tier,
            use_chat_template=False,
            max_tokens=max_tokens,
            num_draft_tokens=4,
            use_speculative=True,
            stop_line=stop_line,
            verbose=False,
        )
        clean_code = trim_fim_overlap(raw_out, suffix)

        applied = False
        if file_path and arguments.get("apply_in_place"):
            Path(file_path).write_text(prefix + clean_code + suffix, encoding="utf-8")
            applied = True

        return {
            "tier": tier,
            "infilled_code": clean_code,
            "applied_in_place": applied,
            "file_path": file_path,
            "metrics": metrics,
        }

    if name == "mlx_surgeon_format":
        prompt = arguments.get("prompt")
        if not prompt:
            raise ValueError("'prompt' is required for mlx_surgeon_format")
        tier = arguments.get("tier", "fast")
        max_tokens = int(arguments.get("max_tokens", 512))

        output, metrics = run_generation(
            prompt,
            tier=tier,
            use_chat_template=True,
            max_tokens=max_tokens,
            num_draft_tokens=4,
            use_speculative=True,
            verbose=False,
        )
        return {
            "tier": tier,
            "output": output.strip(),
            "metrics": metrics,
        }

    raise ValueError(f"Unknown tool: {name}")


def main() -> int:
    """Run the JSON-RPC 2.0 Stdio MCP server loop."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "mlx-code-surgeon",
                        "version": "0.1.0",
                    },
                },
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method.startswith("notifications/"):
            continue

        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS_SCHEMA},
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments") or {}
            try:
                result_data = handle_tool_call(tool_name, tool_args)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result_data, indent=2),
                            }
                        ],
                        "isError": False,
                    },
                }
            except Exception as exc:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({"error": str(exc)}, indent=2),
                            }
                        ],
                        "isError": True,
                    },
                }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        elif req_id is not None:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
