from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_ssd import ContextConfig, benchmark_pair, write_pair_report


DEFAULT_PROMPT = (
    "Explain how speculative decoding, KV cache reuse, attention head layout, and CPU/GPU overlap "
    "affect local LLM inference throughput on a workstation."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a standalone Vulkan-backed SSD benchmark.")
    parser.add_argument("--draft-model", required=True, type=Path)
    parser.add_argument("--target-model", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--draft-k", type=int, default=4)
    parser.add_argument("--branch-count", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-batch", type=int, default=256)
    parser.add_argument("--draft-gpu-layers", type=int, default=0)
    parser.add_argument("--target-gpu-layers", type=int, default=-1)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/vulkan_ssd_benchmark.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/vulkan_ssd_benchmark.md"))
    args = parser.parse_args()

    config = ContextConfig(n_ctx=args.n_ctx, n_batch=args.n_batch, n_ubatch=args.n_batch)
    result = benchmark_pair(
        args.draft_model,
        args.target_model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        draft_k=args.draft_k,
        branch_count=args.branch_count,
        draft_gpu_layers=args.draft_gpu_layers,
        target_gpu_layers=args.target_gpu_layers,
        context=config,
    )
    write_pair_report(result, output_json=args.output_json, output_md=args.output_md)
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote markdown report to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
