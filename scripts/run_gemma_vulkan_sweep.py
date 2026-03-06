from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_ssd import ContextConfig, benchmark_pair


DEFAULT_PROMPT = (
    "Explain how speculative decoding, KV cache reuse, attention head layout, and CPU/GPU overlap "
    "affect local LLM inference throughput on a workstation."
)

DEFAULT_MODELS_DIR = Path.home() / ".lmstudio" / "models" / "lmstudio-community"
DEFAULT_PAIRS = [
    {
        "label": "gemma-1b-to-4b",
        "draft_model": DEFAULT_MODELS_DIR / "gemma-3-1B-it-QAT-GGUF" / "gemma-3-1B-it-QAT-Q4_0.gguf",
        "target_model": DEFAULT_MODELS_DIR / "gemma-3-4b-it-GGUF" / "gemma-3-4b-it-Q4_K_M.gguf",
        "draft_gpu_layers": 0,
        "target_gpu_layers": -1,
    },
    {
        "label": "gemma-4b-to-27b",
        "draft_model": DEFAULT_MODELS_DIR / "gemma-3-4b-it-GGUF" / "gemma-3-4b-it-Q4_K_M.gguf",
        "target_model": DEFAULT_MODELS_DIR / "gemma-3-27B-it-qat-GGUF" / "gemma-3-27B-it-QAT-Q4_0.gguf",
        "draft_gpu_layers": 0,
        "target_gpu_layers": 24,
    },
]


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _make_markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "# Gemma Vulkan Sweep",
        "",
        "| Pair | k | Target GPU layers | AR tok/s | Sync tok/s | Async tok/s | Sync/AR | Async/AR | Sync accept | Async accept | Branch hit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        sync = item["sync_speculative"]
        async_prefetch = item["async_prefetch"]
        target_ar = item["target_ar"]
        lines.append(
            "| {label} | {k} | {target_gpu_layers} | {ar_tps:.2f} | {sync_tps:.2f} | {async_tps:.2f} | {sync_ratio} | {async_ratio} | {sync_accept} | {async_accept} | {branch_hit} |".format(
                label=item["label"],
                k=item["draft_k"],
                target_gpu_layers=item["target_gpu_layers"],
                ar_tps=target_ar["tokens_per_second"],
                sync_tps=sync["tokens_per_second"],
                async_tps=async_prefetch["tokens_per_second"],
                sync_ratio=f"{item['sync_vs_ar']:.2f}x" if item["sync_vs_ar"] is not None else "",
                async_ratio=f"{item['async_vs_ar']:.2f}x" if item["async_vs_ar"] is not None else "",
                sync_accept=f"{sync['acceptance_rate']:.3f}" if sync["acceptance_rate"] is not None else "",
                async_accept=f"{async_prefetch['acceptance_rate']:.3f}" if async_prefetch["acceptance_rate"] is not None else "",
                branch_hit=f"{async_prefetch['branch_hit_rate']:.3f}" if async_prefetch["branch_hit_rate"] is not None else "",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small standalone Vulkan sweep for the Gemma speculative pairs.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--branch-count", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=1024)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/gemma_vulkan_sweep.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/gemma_vulkan_sweep.md"))
    args = parser.parse_args()

    config = ContextConfig(n_ctx=args.n_ctx, n_batch=args.n_batch, n_ubatch=args.n_batch)
    results: list[dict[str, object]] = []

    for pair in DEFAULT_PAIRS:
        for draft_k in args.k_values:
            print(f"Benchmarking {pair['label']} with k={draft_k}")
            benchmark = benchmark_pair(
                pair["draft_model"],
                pair["target_model"],
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                draft_k=draft_k,
                branch_count=args.branch_count,
                draft_gpu_layers=pair["draft_gpu_layers"],
                target_gpu_layers=pair["target_gpu_layers"],
                context=config,
            ).to_dict()
            results.append(
                {
                    "label": pair["label"],
                    "draft_k": draft_k,
                    "branch_count": args.branch_count,
                    "target_gpu_layers": pair["target_gpu_layers"],
                    "draft_model": benchmark["draft_model"],
                    "target_model": benchmark["target_model"],
                    "target_ar": benchmark["target_ar"],
                    "sync_speculative": benchmark["sync_speculative"],
                    "async_prefetch": benchmark["async_prefetch"],
                    "sync_vs_ar": _ratio(
                        benchmark["sync_speculative"]["tokens_per_second"],
                        benchmark["target_ar"]["tokens_per_second"],
                    ),
                    "async_vs_ar": _ratio(
                        benchmark["async_prefetch"]["tokens_per_second"],
                        benchmark["target_ar"]["tokens_per_second"],
                    ),
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output_md.write_text(_make_markdown(results), encoding="utf-8")
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote markdown report to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
