from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_ssd import LMStudioAPI, benchmark_chunked_speculation


DEFAULT_PROMPT = (
    "Explain how multi-head attention, KV head sharing, context length, and concurrency "
    "affect local inference throughput on a workstation. Keep it technical."
)


def make_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Custom SSD-Style Benchmark",
        "",
        f"- Draft model: `{result['draft_model']}`",
        f"- Target model: `{result['target_model']}`",
        f"- Chunk tokens: `{result['chunk_tokens']}`",
        f"- Iterations: `{result['iterations']}`",
        f"- Draft mean tok/s: `{result['draft_mean_tokens_per_second']}`",
        f"- Target mean tok/s: `{result['target_mean_tokens_per_second']}`",
        f"- Mean acceptance ratio: `{result['mean_acceptance_ratio']}`",
        "",
        "| Iteration | Draft tok | Target tok | Accepted tok | Acceptance ratio | Draft s | Target s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result["chunk_results"]:
        lines.append(
            "| {iteration} | {draft_completion_tokens} | {target_completion_tokens} | {accepted_tokens} | {accepted_ratio:.3f} | {draft_elapsed_seconds:.2f} | {target_elapsed_seconds:.2f} |".format(
                **item
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a custom SSD-style chunk-agreement benchmark through LM Studio.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/custom_ssd_benchmark.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/custom_ssd_benchmark.md"))
    args = parser.parse_args()

    api = LMStudioAPI(base_url=args.base_url, timeout=args.timeout)
    models = api.list_models()
    for model in (args.draft_model, args.target_model):
        if model not in models:
            raise SystemExit(f"Model '{model}' is not exposed by LM Studio. Available models: {models}")

    result = benchmark_chunked_speculation(
        api,
        draft_model=args.draft_model,
        target_model=args.target_model,
        prompt=args.prompt,
        chunk_tokens=args.chunk_tokens,
        iterations=args.iterations,
        temperature=args.temperature,
    ).to_dict()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_md.write_text(make_markdown(result), encoding="utf-8")
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote markdown report to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
