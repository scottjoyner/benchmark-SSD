from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .vulkan_llama import (
    ContextConfig,
    ContextSnapshot,
    ModelCompatibilityError,
    get_backend,
)


TOKENIZER_PROBE_STRINGS = [
    "Speculative decoding on Windows Vulkan.",
    "This benchmark checks tokenizer compatibility.",
    "Attention heads, KV heads, and concurrency matter.",
]


@dataclass(slots=True)
class BenchmarkMetrics:
    algorithm: str
    prompt: str
    output_text: str
    output_tokens: int
    prompt_seconds: float
    generation_seconds: float
    total_seconds: float
    tokens_per_second: float
    rounds: int | None = None
    drafted_tokens: int | None = None
    accepted_tokens: int | None = None
    acceptance_rate: float | None = None
    branch_hits: int | None = None
    branch_hit_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PairBenchmarkResult:
    draft_model: str
    target_model: str
    target_ar: BenchmarkMetrics
    sync_speculative: BenchmarkMetrics
    async_prefetch: BenchmarkMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "draft_model": self.draft_model,
            "target_model": self.target_model,
            "target_ar": self.target_ar.to_dict(),
            "sync_speculative": self.sync_speculative.to_dict(),
            "async_prefetch": self.async_prefetch.to_dict(),
        }


@dataclass(slots=True)
class _PrefetchedBranch:
    bonus_token: int
    start_snapshot: ContextSnapshot
    end_snapshot: ContextSnapshot
    tokens: list[int]


def _ensure_tokenizer_compatibility(draft_model, target_model) -> None:
    for probe in TOKENIZER_PROBE_STRINGS:
        if draft_model.tokenize(probe, add_special=True) != target_model.tokenize(probe, add_special=True):
            raise ModelCompatibilityError(
                f"Tokenizer mismatch between '{draft_model.path.name}' and '{target_model.path.name}'"
            )


def _format_output(model, tokens: list[int]) -> str:
    return model.detokenize(tokens, remove_special=False, unparse_special=True)


def _generate_chunk(ctx, token_budget: int) -> list[int]:
    tokens: list[int] = []
    for _ in range(token_budget):
        token = ctx.sample_greedy()
        tokens.append(token)
        ctx.decode([token])
        if ctx.model.is_eog(token):
            break
    return tokens


def run_autoregressive(
    model_path: str | Path,
    *,
    prompt: str,
    max_new_tokens: int,
    n_gpu_layers: int = -1,
    context: ContextConfig | None = None,
) -> BenchmarkMetrics:
    backend = get_backend()
    generated: list[int] = []

    with backend.load_model(model_path, n_gpu_layers=n_gpu_layers) as model:
        with model.create_context(context) as ctx:
            prompt_tokens = model.tokenize(prompt, add_special=True)
            prompt_start = time.perf_counter()
            ctx.decode(prompt_tokens)
            prompt_seconds = time.perf_counter() - prompt_start

            generation_start = time.perf_counter()
            for _ in range(max_new_tokens):
                token = ctx.sample_greedy()
                generated.append(token)
                ctx.decode([token])
                if model.is_eog(token):
                    break
            generation_seconds = time.perf_counter() - generation_start

        output_text = _format_output(model, generated)

    return BenchmarkMetrics(
        algorithm="autoregressive",
        prompt=prompt,
        output_text=output_text,
        output_tokens=len(generated),
        prompt_seconds=prompt_seconds,
        generation_seconds=generation_seconds,
        total_seconds=prompt_seconds + generation_seconds,
        tokens_per_second=(len(generated) / generation_seconds) if generation_seconds > 0 else 0.0,
    )


def run_sync_speculative(
    draft_model_path: str | Path,
    target_model_path: str | Path,
    *,
    prompt: str,
    max_new_tokens: int,
    draft_k: int,
    draft_gpu_layers: int = 0,
    target_gpu_layers: int = -1,
    context: ContextConfig | None = None,
) -> BenchmarkMetrics:
    backend = get_backend()
    generated: list[int] = []
    drafted_tokens = 0
    accepted_tokens = 0
    rounds = 0

    with backend.load_model(draft_model_path, n_gpu_layers=draft_gpu_layers) as draft_model:
        with backend.load_model(target_model_path, n_gpu_layers=target_gpu_layers) as target_model:
            _ensure_tokenizer_compatibility(draft_model, target_model)
            prompt_tokens = target_model.tokenize(prompt, add_special=True)

            with draft_model.create_context(context) as draft_ctx, target_model.create_context(context) as target_ctx:
                prompt_start = time.perf_counter()
                draft_ctx.decode(prompt_tokens)
                target_ctx.decode(prompt_tokens)
                prompt_seconds = time.perf_counter() - prompt_start

                generation_start = time.perf_counter()
                while len(generated) < max_new_tokens:
                    rounds += 1
                    draft_base = draft_ctx.snapshot()
                    target_base = target_ctx.snapshot()

                    chunk = _generate_chunk(draft_ctx, draft_k)
                    if not chunk:
                        break
                    drafted_tokens += len(chunk)

                    prefix_prediction = target_ctx.sample_greedy()
                    target_ctx.decode(chunk, logits_all=True)

                    matched = 0
                    if prefix_prediction == chunk[0]:
                        matched = 1
                        for idx in range(1, len(chunk)):
                            predicted = target_ctx.argmax_for(idx - 1)
                            if predicted != chunk[idx]:
                                break
                            matched += 1

                    accepted = chunk[:matched]
                    accepted_tokens += matched

                    if matched == len(chunk):
                        next_true = target_ctx.argmax_for(len(chunk) - 1)
                    else:
                        next_true = prefix_prediction if matched == 0 else target_ctx.argmax_for(matched - 1)
                        target_ctx.restore(target_base)
                        if accepted:
                            target_ctx.decode(accepted)

                    generated.extend(accepted)
                    if len(generated) >= max_new_tokens:
                        break

                    generated.append(next_true)
                    target_ctx.decode([next_true])

                    draft_ctx.restore(draft_base)
                    if accepted:
                        draft_ctx.decode(accepted)
                    draft_ctx.decode([next_true])

                    if target_model.is_eog(next_true):
                        break

                generation_seconds = time.perf_counter() - generation_start

            output_text = _format_output(target_model, generated[:max_new_tokens])

    return BenchmarkMetrics(
        algorithm="sync_speculative",
        prompt=prompt,
        output_text=output_text,
        output_tokens=min(len(generated), max_new_tokens),
        prompt_seconds=prompt_seconds,
        generation_seconds=generation_seconds,
        total_seconds=prompt_seconds + generation_seconds,
        tokens_per_second=(min(len(generated), max_new_tokens) / generation_seconds) if generation_seconds > 0 else 0.0,
        rounds=rounds,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        acceptance_rate=(accepted_tokens / drafted_tokens) if drafted_tokens else 0.0,
    )


def _prefetch_accept_all_branches(
    ctx,
    base_snapshot: ContextSnapshot,
    branch_tokens: list[int],
    *,
    branch_count: int,
    draft_k: int,
) -> dict[int, _PrefetchedBranch]:
    ctx.restore(base_snapshot)
    ctx.decode(branch_tokens)
    candidate_tokens = ctx.top_tokens_for(ctx._last_logits_index or 0, branch_count)
    prefetched: dict[int, _PrefetchedBranch] = {}
    for bonus_token in candidate_tokens:
        ctx.restore(base_snapshot)
        ctx.decode(branch_tokens)
        ctx.decode([bonus_token])
        start_snapshot = ctx.snapshot()
        next_chunk = _generate_chunk(ctx, draft_k)
        end_snapshot = ctx.snapshot()
        prefetched[bonus_token] = _PrefetchedBranch(
            bonus_token=bonus_token,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            tokens=next_chunk,
        )
    return prefetched


def run_async_prefetch(
    draft_model_path: str | Path,
    target_model_path: str | Path,
    *,
    prompt: str,
    max_new_tokens: int,
    draft_k: int,
    branch_count: int = 2,
    draft_gpu_layers: int = 0,
    target_gpu_layers: int = -1,
    context: ContextConfig | None = None,
) -> BenchmarkMetrics:
    backend = get_backend()
    generated: list[int] = []
    drafted_tokens = 0
    accepted_tokens = 0
    rounds = 0
    branch_hits = 0
    prefetched_branch: _PrefetchedBranch | None = None

    with backend.load_model(draft_model_path, n_gpu_layers=draft_gpu_layers) as draft_model:
        with backend.load_model(target_model_path, n_gpu_layers=target_gpu_layers) as target_model:
            _ensure_tokenizer_compatibility(draft_model, target_model)
            prompt_tokens = target_model.tokenize(prompt, add_special=True)

            with (
                draft_model.create_context(context) as draft_ctx,
                draft_model.create_context(context) as prefetch_ctx,
                target_model.create_context(context) as target_ctx,
                ThreadPoolExecutor(max_workers=1) as executor,
            ):
                prompt_start = time.perf_counter()
                draft_ctx.decode(prompt_tokens)
                prefetch_ctx.decode(prompt_tokens)
                target_ctx.decode(prompt_tokens)
                prompt_seconds = time.perf_counter() - prompt_start

                generation_start = time.perf_counter()
                pending_prefetch: Future[dict[int, _PrefetchedBranch]] | None = None

                while len(generated) < max_new_tokens:
                    rounds += 1
                    draft_base = draft_ctx.snapshot()
                    target_base = target_ctx.snapshot()

                    if prefetched_branch is not None:
                        chunk = list(prefetched_branch.tokens)
                        draft_ctx.restore(prefetched_branch.end_snapshot)
                        prefetched_branch = None
                    else:
                        chunk = _generate_chunk(draft_ctx, draft_k)
                    if not chunk:
                        break
                    drafted_tokens += len(chunk)

                    pending_prefetch = executor.submit(
                        _prefetch_accept_all_branches,
                        prefetch_ctx,
                        draft_base,
                        chunk,
                        branch_count=branch_count,
                        draft_k=draft_k,
                    )

                    prefix_prediction = target_ctx.sample_greedy()
                    target_ctx.decode(chunk, logits_all=True)

                    matched = 0
                    if prefix_prediction == chunk[0]:
                        matched = 1
                        for idx in range(1, len(chunk)):
                            predicted = target_ctx.argmax_for(idx - 1)
                            if predicted != chunk[idx]:
                                break
                            matched += 1

                    accepted = chunk[:matched]
                    accepted_tokens += matched

                    if matched == len(chunk):
                        next_true = target_ctx.argmax_for(len(chunk) - 1)
                    else:
                        next_true = prefix_prediction if matched == 0 else target_ctx.argmax_for(matched - 1)
                        target_ctx.restore(target_base)
                        if accepted:
                            target_ctx.decode(accepted)

                    generated.extend(accepted)
                    if len(generated) >= max_new_tokens:
                        break

                    generated.append(next_true)
                    target_ctx.decode([next_true])

                    draft_ctx.restore(draft_base)
                    if accepted:
                        draft_ctx.decode(accepted)
                    draft_ctx.decode([next_true])

                    if matched == len(chunk) and pending_prefetch is not None:
                        branches = pending_prefetch.result()
                        chosen = branches.get(next_true)
                        if chosen is not None:
                            prefetched_branch = chosen
                            branch_hits += 1

                    if target_model.is_eog(next_true):
                        break

                generation_seconds = time.perf_counter() - generation_start

            output_text = _format_output(target_model, generated[:max_new_tokens])

    return BenchmarkMetrics(
        algorithm="async_prefetch",
        prompt=prompt,
        output_text=output_text,
        output_tokens=min(len(generated), max_new_tokens),
        prompt_seconds=prompt_seconds,
        generation_seconds=generation_seconds,
        total_seconds=prompt_seconds + generation_seconds,
        tokens_per_second=(min(len(generated), max_new_tokens) / generation_seconds) if generation_seconds > 0 else 0.0,
        rounds=rounds,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        acceptance_rate=(accepted_tokens / drafted_tokens) if drafted_tokens else 0.0,
        branch_hits=branch_hits,
        branch_hit_rate=(branch_hits / rounds) if rounds else 0.0,
    )


def benchmark_pair(
    draft_model_path: str | Path,
    target_model_path: str | Path,
    *,
    prompt: str,
    max_new_tokens: int,
    draft_k: int,
    branch_count: int = 2,
    draft_gpu_layers: int = 0,
    target_gpu_layers: int = -1,
    context: ContextConfig | None = None,
) -> PairBenchmarkResult:
    target_ar = run_autoregressive(
        target_model_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        n_gpu_layers=target_gpu_layers,
        context=context,
    )
    sync = run_sync_speculative(
        draft_model_path,
        target_model_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_k=draft_k,
        draft_gpu_layers=draft_gpu_layers,
        target_gpu_layers=target_gpu_layers,
        context=context,
    )
    async_prefetch = run_async_prefetch(
        draft_model_path,
        target_model_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        draft_k=draft_k,
        branch_count=branch_count,
        draft_gpu_layers=draft_gpu_layers,
        target_gpu_layers=target_gpu_layers,
        context=context,
    )
    return PairBenchmarkResult(
        draft_model=str(Path(draft_model_path).resolve()),
        target_model=str(Path(target_model_path).resolve()),
        target_ar=target_ar,
        sync_speculative=sync,
        async_prefetch=async_prefetch,
    )


def write_pair_report(result: PairBenchmarkResult, *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    output_md.write_text(_make_markdown(result), encoding="utf-8")


def _make_markdown(result: PairBenchmarkResult) -> str:
    rows = [result.target_ar, result.sync_speculative, result.async_prefetch]
    lines = [
        "# Standalone Vulkan SSD Benchmark",
        "",
        f"- Draft model: `{result.draft_model}`",
        f"- Target model: `{result.target_model}`",
        "",
        "| Algorithm | Output tok | Tok/s | Prompt s | Gen s | Acceptance | Branch hit | Rounds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {output_tokens} | {tps:.2f} | {prompt_seconds:.2f} | {generation_seconds:.2f} | {acceptance} | {branch_hit} | {rounds} |".format(
                algorithm=row.algorithm,
                output_tokens=row.output_tokens,
                tps=row.tokens_per_second,
                prompt_seconds=row.prompt_seconds,
                generation_seconds=row.generation_seconds,
                acceptance=f"{row.acceptance_rate:.3f}" if row.acceptance_rate is not None else "",
                branch_hit=f"{row.branch_hit_rate:.3f}" if row.branch_hit_rate is not None else "",
                rounds=row.rounds or "",
            )
        )
    return "\n".join(lines) + "\n"
