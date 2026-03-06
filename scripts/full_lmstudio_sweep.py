from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:1234"
MODEL_INDEX_CACHE = Path(r"C:\Users\AMD\.lmstudio\.internal\model-index-cache.json")
OUTPUT_JSON = Path("artifacts/full_lmstudio_sweep.json")
OUTPUT_MD = Path("reports/lmstudio_full_matrix_2026-03-05.md")

TEXT_PROMPT = (
    "Explain how attention heads, KV heads, quantization, and context length can change local inference "
    "performance on a workstation. Keep the answer concise and technical."
)
EMBED_INPUT = (
    "Benchmarking local inference requires separating cold-start load cost from steady-state decode speed."
)


def get_json(url: str, timeout: int) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


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


def normalize_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def load_model_metadata(index_cache: Path) -> list[dict[str, object]]:
    payload = json.loads(index_cache.read_text(encoding="utf-8"))
    result: list[dict[str, object]] = []
    for model in payload.get("models", []):
        gguf = ((model.get("metadata") or {}).get("gguf")) or {}
        entry_point = model.get("entryPoint") or {}
        key = str(model.get("containingDirSubpath") or "")
        display_name = str(model.get("displayName") or "")
        indexed = str(model.get("indexedModelIdentifier") or "")
        result.append(
            {
                "lookup_keys": [
                    normalize_key(key),
                    normalize_key(display_name),
                    normalize_key(indexed.split("@")[0]),
                    normalize_key(indexed),
                    normalize_key(str(entry_point.get("filename") or "")),
                ],
            "domain": model.get("domain"),
            "display_name": display_name,
            "size_bytes": entry_point.get("sizeBytes"),
            "quant": ((model.get("quant") or {}).get("name")),
            "arch": gguf.get("arch"),
            "params": gguf.get("parameters"),
            "num_attention_heads": gguf.get("numAttentionHeads"),
            "num_kv_heads": gguf.get("numKeyValueHeads"),
            "num_layers": gguf.get("numLayers"),
            "context_length": gguf.get("contextLength"),
            "embedding_length": gguf.get("embeddingLength"),
            "num_experts": gguf.get("numExperts"),
            "default_num_experts": gguf.get("defaultNumExperts"),
            "vision": "vl" in str(key).lower() or "vision" in str(key).lower() or ".6v" in str(key).lower(),
            }
        )
    return result


def resolve_metadata(model_id: str, metadata_list: list[dict[str, object]]) -> dict[str, object]:
    target = normalize_key(model_id)
    best: dict[str, object] | None = None
    best_score = -1
    for item in metadata_list:
        keys = item.get("lookup_keys", [])
        score = 0
        for key in keys:
            if not key:
                continue
            if key == target:
                score = max(score, 100)
            elif target in key or key in target:
                score = max(score, min(len(key), len(target)))
        if score > best_score:
            best = item
            best_score = score
    if best is None:
        return {"domain": "unknown"}
    return {k: v for k, v in best.items() if k != "lookup_keys"}


def classify_model(model_id: str, metadata: dict[str, object]) -> str:
    domain = str(metadata.get("domain") or "")
    if model_id.startswith("text-embedding") or domain == "embedding":
        return "embedding"
    if metadata.get("vision"):
        return "vision"
    if metadata.get("num_experts"):
        return "moe"
    return "text"


def run_chat_request(base_url: str, model: str, max_tokens: int, temperature: float, timeout: int) -> dict[str, object]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEXT_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    start = time.perf_counter()
    response = post_json(f"{base_url}/v1/chat/completions", payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    total_tokens = int(usage.get("total_tokens", completion_tokens + prompt_tokens))
    return {
        "elapsed_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": (completion_tokens / elapsed) if elapsed > 0 else None,
    }


def run_embedding_request(base_url: str, model: str, timeout: int) -> dict[str, object]:
    payload = {"model": model, "input": EMBED_INPUT}
    start = time.perf_counter()
    response = post_json(f"{base_url}/v1/embeddings", payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    usage = response.get("usage", {})
    total_tokens = int(usage.get("total_tokens", 0))
    emb = response.get("data", [{}])[0].get("embedding", [])
    return {
        "elapsed_seconds": elapsed,
        "total_tokens": total_tokens,
        "embedding_dimensions_returned": len(emb),
        "tokens_per_second": (total_tokens / elapsed) if elapsed > 0 and total_tokens else None,
        "requests_per_second": (1 / elapsed) if elapsed > 0 else None,
    }


def run_parallel(kind: str, base_url: str, model: str, max_tokens: int, temperature: float, timeout: int, concurrency: int) -> dict[str, object]:
    if kind == "embedding":
        fn = lambda: run_embedding_request(base_url, model, timeout)
    else:
        fn = lambda: run_chat_request(base_url, model, max_tokens, temperature, timeout)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fn) for _ in range(concurrency)]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - start

    total_completion_tokens = sum(int(item.get("completion_tokens", 0)) for item in results)
    total_tokens = sum(int(item.get("total_tokens", 0)) for item in results)
    return {
        "concurrency": concurrency,
        "wall_seconds": wall,
        "mean_elapsed_seconds": statistics.mean(item["elapsed_seconds"] for item in results),
        "aggregate_completion_tokens_per_second": (total_completion_tokens / wall) if total_completion_tokens else None,
        "aggregate_total_tokens_per_second": (total_tokens / wall) if total_tokens else None,
        "aggregate_requests_per_second": concurrency / wall if wall > 0 else None,
    }


def summarize_num_kv_heads(value: object) -> object:
    if isinstance(value, list):
        distinct = sorted({int(v) for v in value})
        return distinct
    return value


def benchmark_model(model_id: str, metadata: dict[str, object], base_url: str, timeout: int, max_tokens: int, temperature: float, concurrency: int) -> dict[str, object]:
    kind = classify_model(model_id, metadata)
    result: dict[str, object] = {
        "model": model_id,
        "kind": kind,
        "metadata": metadata,
    }
    if kind == "embedding":
        cold = run_embedding_request(base_url, model_id, timeout)
        warm = run_embedding_request(base_url, model_id, timeout)
    else:
        cold = run_chat_request(base_url, model_id, max_tokens, temperature, timeout)
        warm = run_chat_request(base_url, model_id, max_tokens, temperature, timeout)

    parallel = run_parallel(kind, base_url, model_id, max_tokens, temperature, timeout, concurrency)
    result["cold"] = cold
    result["warm"] = warm
    result["parallel"] = parallel
    return result


def make_markdown(results: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# LM Studio Full Matrix - 2026-03-05",
        "",
        "Cold rows are first-request timings after model switch. Warm rows are a second single request against the same model. Parallel rows use 4 simultaneous requests.",
        "",
        "| Model | Kind | Params | Heads | KV Heads | Layers | Context | Quant | Size GiB | Cold s | Warm tok/s | Parallel agg tok/s | Parallel req/s |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        meta = item["metadata"]
        size_bytes = meta.get("size_bytes") or 0
        size_gib = (size_bytes / (1024 ** 3)) if size_bytes else None
        warm_tps = item["warm"].get("tokens_per_second")
        par_tps = item["parallel"].get("aggregate_completion_tokens_per_second")
        if item["kind"] == "embedding":
            par_tps = item["parallel"].get("aggregate_total_tokens_per_second")
        lines.append(
            "| {model} | {kind} | {params} | {heads} | {kv} | {layers} | {ctx} | {quant} | {size} | {cold:.2f} | {warm} | {par} | {rps:.2f} |".format(
                model=item["model"],
                kind=item["kind"],
                params=meta.get("params") or "",
                heads=meta.get("num_attention_heads") or "",
                kv=summarize_num_kv_heads(meta.get("num_kv_heads")),
                layers=meta.get("num_layers") or "",
                ctx=meta.get("context_length") or "",
                quant=meta.get("quant") or "",
                size=f"{size_gib:.2f}" if size_gib is not None else "",
                cold=item["cold"]["elapsed_seconds"],
                warm=f"{warm_tps:.2f}" if isinstance(warm_tps, float) else "",
                par=f"{par_tps:.2f}" if isinstance(par_tps, float) else "",
                rps=item["parallel"]["aggregate_requests_per_second"],
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all models exposed by LM Studio and generate a performance matrix.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model-index-cache", type=Path, default=MODEL_INDEX_CACHE)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_models = get_json(f"{args.base_url}/v1/models", args.timeout).get("data", [])
    model_ids = [entry["id"] for entry in api_models]
    if args.models:
        wanted = set(args.models)
        model_ids = [model_id for model_id in model_ids if model_id in wanted]
    if args.limit is not None:
        model_ids = model_ids[:args.limit]
    metadata_list = load_model_metadata(args.model_index_cache)

    results: list[dict[str, object]] = []
    for model_id in model_ids:
        metadata = resolve_metadata(model_id, metadata_list)
        print(f"Benchmarking {model_id}...")
        try:
            result = benchmark_model(
                model_id=model_id,
                metadata=metadata,
                base_url=args.base_url,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
            )
        except Exception as exc:
            result = {
                "model": model_id,
                "kind": classify_model(model_id, metadata),
                "metadata": metadata,
                "error": str(exc),
            }
        results.append(result)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    successful = [item for item in results if "error" not in item]
    make_markdown(successful, args.output_md)
    print(f"Wrote JSON results to {args.output_json}")
    print(f"Wrote markdown matrix to {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
