from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:1234"
DEFAULT_PROMPTS = Path("prompts/benchmark_prompts.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/lmstudio-benchmarks")


def load_prompts(path: Path, samples: int | None) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    if samples is not None:
        prompts = prompts[:samples]
    return prompts


def post_json(url: str, payload: dict[str, object], timeout: int) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_models(base_url: str, timeout: int) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["id"] for item in payload.get("data", [])]


def benchmark_prompt(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> dict[str, object]:
    start = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    response = post_json(f"{base_url}/v1/chat/completions", payload, timeout=timeout)
    elapsed = time.perf_counter() - start

    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    total_tokens = int(usage.get("total_tokens", completion_tokens + prompt_tokens))
    tps = completion_tokens / elapsed if elapsed > 0 else None
    text = response["choices"][0]["message"]["content"]
    return {
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": tps,
        "response_preview": text[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a loaded LM Studio model through its OpenAI-compatible API.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    try:
        available_models = get_models(args.base_url, args.timeout)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"LM Studio API is not reachable at {args.base_url}. Start LM Studio's local server first. Details: {exc}"
        )

    if args.model not in available_models:
        raise SystemExit(
            f"Model '{args.model}' is not currently exposed by LM Studio. Available models: {available_models}"
        )

    prompts = load_prompts(args.prompt_file, args.samples)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompt_file}")

    results: list[dict[str, object]] = []
    for prompt in prompts:
        metrics = benchmark_prompt(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt["prompt"],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        results.append({"id": prompt.get("id"), "prompt": prompt["prompt"], "metrics": metrics})
        tps = metrics["tokens_per_second"]
        tps_text = f"{tps:.2f}" if isinstance(tps, float) else "n/a"
        print(f"{prompt.get('id', 'prompt')}: {metrics['elapsed_seconds']:.2f}s, {tps_text} tok/s")

    elapsed_values = [item["metrics"]["elapsed_seconds"] for item in results]
    tps_values = [
        item["metrics"]["tokens_per_second"]
        for item in results
        if isinstance(item["metrics"]["tokens_per_second"], float)
    ]
    completion_tokens = sum(int(item["metrics"]["completion_tokens"]) for item in results)

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "prompt_file": str(args.prompt_file),
        "samples": len(results),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "aggregate": {
            "total_completion_tokens": completion_tokens,
            "mean_elapsed_seconds": statistics.mean(elapsed_values),
            "mean_tokens_per_second": statistics.mean(tps_values) if tps_values else None,
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "__").replace("\\", "__")
    output_path = args.output_dir / f"{safe_name}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Wrote benchmark report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
