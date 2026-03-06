# Gemma Part 2 Benchmark

## Scope

- Draft model: `google/gemma-3-4b`
- Target model: `google/gemma-3-27b`
- Runtime: LM Studio local server on `http://127.0.0.1:1234`
- Goal: check whether the loaded Gemma pair overlaps well in LM Studio and whether endpoint-level output agreement is strong enough to make the pair interesting

## Results

### Chunk agreement

Source: `artifacts/gemma_pair_custom_ssd.json`

- Draft mean throughput: `18.09 tok/s`
- Target mean throughput: `4.50 tok/s`
- Mean acceptance ratio: `0.000`

Across all three 64-token iterations, accepted prefix length was `0`. The 4B and 27B models answered the same prompt in different surface forms from the start, so HTTP-level chunk agreement was zero.

### Sequential vs concurrent pair execution

Source: `artifacts/gemma_pair_concurrency.json`

| Mode | Mean wall s | Draft tok/s | Target tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 16.28 | 19.84 | 4.90 | 0.000 |
| Concurrent | 14.34 | 16.89 | 4.47 | 0.000 |

Concurrent wall-clock time improved by `1.14x` versus sequential execution, which means LM Studio is overlapping work for the loaded Gemma pair. As with the LFM pair, per-model throughput dropped under overlap.

## Important Interpretation

This LM Studio endpoint result should not be confused with the earlier standalone exact Vulkan benchmark.

- At the LM Studio HTTP layer, acceptance was `0.000` because the measurement compares full generated text chunks and those chunks diverged immediately in wording and formatting.
- In the standalone exact Vulkan run, the same Gemma family had non-zero token-level acceptance because verification happened inside the inference loop rather than across two separate completed responses.

So the correct conclusion is:

- LM Studio proves the pair can overlap.
- LM Studio does **not** provide a faithful acceptance signal for exact speculative decoding.
- For Gemma, endpoint-level chunk agreement is too weak to use as a proxy for exact SSD quality.

## Conclusion

This pair is better than LFM for exact speculative work at the engine level, but LM Studio's HTTP endpoint is still the wrong layer for measuring that property directly.

- Good news: the Gemma pair overlaps in LM Studio and shows a modest wall-clock gain under concurrent execution.
- Bad news: HTTP-level agreement remains `0.000`, so endpoint chunk matching is not a useful evaluation method for this pair.
- Practical implication: use the standalone Vulkan path for exact Gemma speculative experiments, and use the LM Studio pair run only as a concurrency/serving test.

## Files

- `reports/gemma_pair_custom_ssd.md`
- `reports/gemma_pair_concurrency.md`
- `artifacts/gemma_pair_custom_ssd.json`
- `artifacts/gemma_pair_concurrency.json`
- `reports/vulkan_standalone_analysis_2026-03-06.md`
