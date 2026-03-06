from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "google/gemma-3-1b",
    "google/gemma-3-4b",
    "mistralai/mistral-7b-instruct-v0.3",
    "deepseek/deepseek-r1-0528-qwen3-8b",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small LM Studio benchmark matrix.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-file", type=Path, default=Path("prompts/benchmark_prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lmstudio-matrix-summary.json"))
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    for model in args.models:
        command = [
            sys.executable,
            str(Path("scripts/benchmark_lmstudio_openai.py")),
            "--model",
            model,
            "--samples",
            str(args.samples),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--prompt-file",
            str(args.prompt_file),
        ]
        print(f"Running {model}...")
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        print(proc.stdout, end="")
        if proc.returncode != 0:
            print(proc.stderr, end="", file=sys.stderr)
            summaries.append({"model": model, "ok": False, "error": proc.stdout.strip() or proc.stderr.strip()})
            continue

        safe_name = model.replace("/", "__").replace("\\", "__")
        report_path = Path("artifacts/lmstudio-benchmarks") / f"{safe_name}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        aggregate = report["aggregate"]
        summaries.append(
            {
                "model": model,
                "ok": True,
                "mean_elapsed_seconds": aggregate["mean_elapsed_seconds"],
                "mean_tokens_per_second": aggregate["mean_tokens_per_second"],
                "total_completion_tokens": aggregate["total_completion_tokens"],
                "report_path": str(report_path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Wrote matrix summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
