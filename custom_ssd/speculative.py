from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from .lmstudio_api import CompletionResult, LMStudioAPI


def _normalize_tokens(text: str) -> list[str]:
    return text.split()


def _longest_common_prefix(left: list[str], right: list[str]) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


@dataclass(slots=True)
class ChunkAgreementResult:
    iteration: int
    draft_text: str
    target_text: str
    draft_completion_tokens: int
    target_completion_tokens: int
    accepted_tokens: int
    accepted_ratio: float
    draft_elapsed_seconds: float
    target_elapsed_seconds: float


@dataclass(slots=True)
class SpeculativeBenchmarkResult:
    draft_model: str
    target_model: str
    prompt: str
    chunk_tokens: int
    iterations: int
    draft_mean_tokens_per_second: float | None
    target_mean_tokens_per_second: float | None
    mean_acceptance_ratio: float | None
    chunk_results: list[ChunkAgreementResult]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["chunk_results"] = [asdict(item) for item in self.chunk_results]
        return payload


def _tokens_per_second(result: CompletionResult) -> float | None:
    if result.elapsed_seconds <= 0 or result.completion_tokens <= 0:
        return None
    return result.completion_tokens / result.elapsed_seconds


def benchmark_chunked_speculation(
    api: LMStudioAPI,
    *,
    draft_model: str,
    target_model: str,
    prompt: str,
    chunk_tokens: int,
    iterations: int,
    temperature: float = 0.0,
) -> SpeculativeBenchmarkResult:
    draft_tps_values: list[float] = []
    target_tps_values: list[float] = []
    acceptance_values: list[float] = []
    chunk_results: list[ChunkAgreementResult] = []

    rolling_prompt = prompt
    for iteration in range(iterations):
        draft = api.chat_completion(
            draft_model,
            rolling_prompt,
            max_tokens=chunk_tokens,
            temperature=temperature,
        )
        target = api.chat_completion(
            target_model,
            rolling_prompt,
            max_tokens=chunk_tokens,
            temperature=temperature,
        )

        draft_tokens = _normalize_tokens(draft.text)
        target_tokens = _normalize_tokens(target.text)
        accepted = _longest_common_prefix(draft_tokens, target_tokens)
        denominator = max(len(draft_tokens), len(target_tokens), 1)
        accepted_ratio = accepted / denominator

        draft_tps = _tokens_per_second(draft)
        target_tps = _tokens_per_second(target)
        if draft_tps is not None:
            draft_tps_values.append(draft_tps)
        if target_tps is not None:
            target_tps_values.append(target_tps)
        acceptance_values.append(accepted_ratio)

        chunk_results.append(
            ChunkAgreementResult(
                iteration=iteration,
                draft_text=draft.text,
                target_text=target.text,
                draft_completion_tokens=draft.completion_tokens,
                target_completion_tokens=target.completion_tokens,
                accepted_tokens=accepted,
                accepted_ratio=accepted_ratio,
                draft_elapsed_seconds=draft.elapsed_seconds,
                target_elapsed_seconds=target.elapsed_seconds,
            )
        )

        accepted_text = " ".join(draft_tokens[:accepted]).strip()
        if accepted_text:
            rolling_prompt = f"{rolling_prompt}\n{accepted_text}"

    return SpeculativeBenchmarkResult(
        draft_model=draft_model,
        target_model=target_model,
        prompt=prompt,
        chunk_tokens=chunk_tokens,
        iterations=iterations,
        draft_mean_tokens_per_second=statistics.mean(draft_tps_values) if draft_tps_values else None,
        target_mean_tokens_per_second=statistics.mean(target_tps_values) if target_tps_values else None,
        mean_acceptance_ratio=statistics.mean(acceptance_values) if acceptance_values else None,
        chunk_results=chunk_results,
    )
