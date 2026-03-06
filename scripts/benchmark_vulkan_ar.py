from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_ssd import ContextConfig, run_autoregressive


DEFAULT_PROMPT = (
    "Explain speculative decoding, exact verification, and GPU overlap in technical terms."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a standalone Vulkan-backed autoregressive benchmark.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--gpu-layers", type=int, default=24)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/vulkan_ar_benchmark.json"))
    args = parser.parse_args()

    config = ContextConfig(n_ctx=args.n_ctx, n_batch=args.n_batch, n_ubatch=args.n_batch)
    result = run_autoregressive(
        args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        n_gpu_layers=args.gpu_layers,
        context=config,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote JSON results to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
