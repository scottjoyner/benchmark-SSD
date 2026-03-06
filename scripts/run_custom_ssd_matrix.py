from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_ssd import LMStudioAPI, benchmark_chunked_speculation


DEFAULT_PAIRS = [
    ("google/gemma-3-1b", "google/gemma-3-4b"),
    ("google/gemma-3-4b", "google/gemma-3-27b"),
    ("microsoft/phi-4-mini-reasoning", "microsoft/phi-4-reasoning-plus"),
    ("mistralai/ministral-3-3b", "mistralai/mistral-7b-instruct-v0.3"),
]

DEFAULT_PROMPT = (
    "Explain how multi-head attention, KV head sharing, quantization, and parallel request execution "
    "change local inference throughput on a workstation. Keep the answer concise and technical."
)


def make_markdown(results: list[dict[str, object]]) -> str:
    lines = [
        "# Custom SSD Matrix",
        "",
        "| Draft | Target | Chunk | Iter | Draft tok/s | Target tok/s | Acceptance ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        if "error" in item:
            lines.append(f"| `{item['draft_model']}` | `{item['target_model']}` |  |  |  |  | error: {item['error']} |")
            continue
        lines.append(
            "| `{draft}` | `{target}` | {chunk} | {iters} | {draft_tps} | {target_tps} | {acceptance} |".format(
                draft=item["draft_model"],
                target=item["target_model"],
                chunk=item["chunk_tokens"],
                iters=item["iterations"],
                draft_tps=f"{item['draft_mean_tokens_per_second']:.2f}" if item["draft_mean_tokens_per_second"] is not None else "",
                target_tps=f"{item['target_mean_tokens_per_second']:.2f}" if item["target_mean_tokens_per_second"] is not None else "",
                acceptance=f"{item['mean_acceptance_ratio']:.3f}" if item["mean_acceptance_ratio"] is not None else "",
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a custom SSD-style matrix against LM Studio models.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/custom_ssd_matrix.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/custom_ssd_matrix.md"))
    args = parser.parse_args()

    api = LMStudioAPI(base_url=args.base_url, timeout=args.timeout)
    models = set(api.list_models())

    results: list[dict[str, object]] = []
    for draft_model, target_model in DEFAULT_PAIRS:
        if draft_model not in models or target_model not in models:
            results.append(
                {
                    "draft_model": draft_model,
                    "target_model": target_model,
                    "error": "model not exposed by LM Studio",
                }
            )
            continue
        print(f"Benchmarking {draft_model} -> {target_model}")
        try:
            result = benchmark_chunked_speculation(
                api,
                draft_model=draft_model,
                target_model=target_model,
                prompt=args.prompt,
                chunk_tokens=args.chunk_tokens,
                iterations=args.iterations,
                temperature=args.temperature,
            ).to_dict()
        except Exception as exc:
            result = {
                "draft_model": draft_model,
                "target_model": target_model,
                "error": str(exc),
            }
        results.append(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.output_md.write_text(make_markdown(results), encoding="utf-8")
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote markdown report to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
