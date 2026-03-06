from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from custom_ssd import LMStudioAPI


DEFAULT_PROMPT = (
    "Explain how multi-head attention, KV head sharing, expert routing, and concurrent execution "
    "affect local inference throughput on a workstation. Keep it technical."
)


def _normalize_tokens(text: str) -> list[str]:
    return text.split()


def _longest_common_prefix(left: list[str], right: list[str]) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


def _tokens_per_second(elapsed_seconds: float, completion_tokens: int) -> float | None:
    if elapsed_seconds <= 0 or completion_tokens <= 0:
        return None
    return completion_tokens / elapsed_seconds


def _run_sequential(
    api: LMStudioAPI,
    *,
    draft_model: str,
    target_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, object]:
    started = time.perf_counter()
    draft = api.chat_completion(draft_model, prompt, max_tokens=max_tokens, temperature=temperature)
    target = api.chat_completion(target_model, prompt, max_tokens=max_tokens, temperature=temperature)
    wall_seconds = time.perf_counter() - started

    draft_tokens = _normalize_tokens(draft.text)
    target_tokens = _normalize_tokens(target.text)
    accepted = _longest_common_prefix(draft_tokens, target_tokens)
    accepted_ratio = accepted / max(len(draft_tokens), len(target_tokens), 1)

    return {
        "mode": "sequential",
        "wall_seconds": wall_seconds,
        "draft_elapsed_seconds": draft.elapsed_seconds,
        "target_elapsed_seconds": target.elapsed_seconds,
        "draft_tokens_per_second": _tokens_per_second(draft.elapsed_seconds, draft.completion_tokens),
        "target_tokens_per_second": _tokens_per_second(target.elapsed_seconds, target.completion_tokens),
        "draft_completion_tokens": draft.completion_tokens,
        "target_completion_tokens": target.completion_tokens,
        "accepted_tokens": accepted,
        "accepted_ratio": accepted_ratio,
        "draft_text": draft.text,
        "target_text": target.text,
    }


def _run_concurrent(
    *,
    base_url: str,
    timeout: int,
    draft_model: str,
    target_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, object]:
    draft_api = LMStudioAPI(base_url=base_url, timeout=timeout)
    target_api = LMStudioAPI(base_url=base_url, timeout=timeout)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        draft_future = pool.submit(
            draft_api.chat_completion,
            draft_model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        target_future = pool.submit(
            target_api.chat_completion,
            target_model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        draft = draft_future.result()
        target = target_future.result()
    wall_seconds = time.perf_counter() - started

    draft_tokens = _normalize_tokens(draft.text)
    target_tokens = _normalize_tokens(target.text)
    accepted = _longest_common_prefix(draft_tokens, target_tokens)
    accepted_ratio = accepted / max(len(draft_tokens), len(target_tokens), 1)

    return {
        "mode": "concurrent",
        "wall_seconds": wall_seconds,
        "draft_elapsed_seconds": draft.elapsed_seconds,
        "target_elapsed_seconds": target.elapsed_seconds,
        "draft_tokens_per_second": _tokens_per_second(draft.elapsed_seconds, draft.completion_tokens),
        "target_tokens_per_second": _tokens_per_second(target.elapsed_seconds, target.completion_tokens),
        "draft_completion_tokens": draft.completion_tokens,
        "target_completion_tokens": target.completion_tokens,
        "accepted_tokens": accepted,
        "accepted_ratio": accepted_ratio,
        "draft_text": draft.text,
        "target_text": target.text,
    }


def _mean(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return statistics.mean(usable) if usable else None


def make_markdown(result: dict[str, object]) -> str:
    sequential = result["sequential_runs"]
    concurrent = result["concurrent_runs"]
    lines = [
        "# LM Studio Pair Concurrency Benchmark",
        "",
        f"- Draft model: `{result['draft_model']}`",
        f"- Target model: `{result['target_model']}`",
        f"- Prompt: `{result['prompt']}`",
        f"- Iterations: `{result['iterations']}`",
        f"- Warmup rounds: `{result['warmup_rounds']}`",
        "",
        "| Mode | Mean wall s | Draft tok/s | Target tok/s | Acceptance | Concurrent gain |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Sequential | {seq_wall:.2f} | {seq_draft_tps} | {seq_target_tps} | {seq_acceptance} |  |".format(
            seq_wall=result["sequential_mean_wall_seconds"],
            seq_draft_tps=f"{result['sequential_mean_draft_tokens_per_second']:.2f}"
            if result["sequential_mean_draft_tokens_per_second"] is not None
            else "",
            seq_target_tps=f"{result['sequential_mean_target_tokens_per_second']:.2f}"
            if result["sequential_mean_target_tokens_per_second"] is not None
            else "",
            seq_acceptance=f"{result['sequential_mean_acceptance_ratio']:.3f}"
            if result["sequential_mean_acceptance_ratio"] is not None
            else "",
        ),
        "| Concurrent | {conc_wall:.2f} | {conc_draft_tps} | {conc_target_tps} | {conc_acceptance} | {gain}x |".format(
            conc_wall=result["concurrent_mean_wall_seconds"],
            conc_draft_tps=f"{result['concurrent_mean_draft_tokens_per_second']:.2f}"
            if result["concurrent_mean_draft_tokens_per_second"] is not None
            else "",
            conc_target_tps=f"{result['concurrent_mean_target_tokens_per_second']:.2f}"
            if result["concurrent_mean_target_tokens_per_second"] is not None
            else "",
            conc_acceptance=f"{result['concurrent_mean_acceptance_ratio']:.3f}"
            if result["concurrent_mean_acceptance_ratio"] is not None
            else "",
            gain=f"{result['concurrent_wall_speedup_vs_sequential']:.2f}"
            if result["concurrent_wall_speedup_vs_sequential"] is not None
            else "",
        ),
        "",
        "## Iterations",
        "",
        "| Mode | Iter | Wall s | Draft s | Target s | Draft tok/s | Target tok/s | Acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for run in sequential + concurrent:
        lines.append(
            "| {mode} | {iteration} | {wall:.2f} | {draft_s:.2f} | {target_s:.2f} | {draft_tps} | {target_tps} | {acceptance:.3f} |".format(
                mode=run["mode"],
                iteration=run["iteration"],
                wall=run["wall_seconds"],
                draft_s=run["draft_elapsed_seconds"],
                target_s=run["target_elapsed_seconds"],
                draft_tps=f"{run['draft_tokens_per_second']:.2f}" if run["draft_tokens_per_second"] is not None else "",
                target_tps=f"{run['target_tokens_per_second']:.2f}" if run["target_tokens_per_second"] is not None else "",
                acceptance=run["accepted_ratio"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sequential vs concurrent LM Studio requests for a draft/target pair.")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/lmstudio_pair_concurrency.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/lmstudio_pair_concurrency.md"))
    args = parser.parse_args()

    api = LMStudioAPI(base_url=args.base_url, timeout=args.timeout)
    models = set(api.list_models())
    for model in (args.draft_model, args.target_model):
        if model not in models:
            raise SystemExit(f"Model '{model}' is not exposed by LM Studio.")

    for _ in range(args.warmup_rounds):
        api.chat_completion(args.draft_model, args.prompt, max_tokens=min(args.max_tokens, 16), temperature=args.temperature)
        api.chat_completion(args.target_model, args.prompt, max_tokens=min(args.max_tokens, 16), temperature=args.temperature)

    sequential_runs: list[dict[str, object]] = []
    concurrent_runs: list[dict[str, object]] = []
    for iteration in range(args.iterations):
        sequential = _run_sequential(
            api,
            draft_model=args.draft_model,
            target_model=args.target_model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        sequential["iteration"] = iteration
        sequential_runs.append(sequential)

        concurrent = _run_concurrent(
            base_url=args.base_url,
            timeout=args.timeout,
            draft_model=args.draft_model,
            target_model=args.target_model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        concurrent["iteration"] = iteration
        concurrent_runs.append(concurrent)

    sequential_mean_wall = statistics.mean(run["wall_seconds"] for run in sequential_runs)
    concurrent_mean_wall = statistics.mean(run["wall_seconds"] for run in concurrent_runs)
    result = {
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "prompt": args.prompt,
        "iterations": args.iterations,
        "warmup_rounds": args.warmup_rounds,
        "sequential_mean_wall_seconds": sequential_mean_wall,
        "concurrent_mean_wall_seconds": concurrent_mean_wall,
        "concurrent_wall_speedup_vs_sequential": (sequential_mean_wall / concurrent_mean_wall) if concurrent_mean_wall > 0 else None,
        "sequential_mean_draft_tokens_per_second": _mean([run["draft_tokens_per_second"] for run in sequential_runs]),
        "sequential_mean_target_tokens_per_second": _mean([run["target_tokens_per_second"] for run in sequential_runs]),
        "concurrent_mean_draft_tokens_per_second": _mean([run["draft_tokens_per_second"] for run in concurrent_runs]),
        "concurrent_mean_target_tokens_per_second": _mean([run["target_tokens_per_second"] for run in concurrent_runs]),
        "sequential_mean_acceptance_ratio": statistics.mean(run["accepted_ratio"] for run in sequential_runs),
        "concurrent_mean_acceptance_ratio": statistics.mean(run["accepted_ratio"] for run in concurrent_runs),
        "sequential_runs": sequential_runs,
        "concurrent_runs": concurrent_runs,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.output_md.write_text(make_markdown(result), encoding="utf-8")
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote markdown report to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
