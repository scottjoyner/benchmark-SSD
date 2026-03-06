# LFM Part 1 Benchmark

## Scope

- Draft model: `liquid/lfm2.5-1.2b`
- Target model: `liquid/lfm2-24b-a2b`
- Runtime: LM Studio local server on `http://127.0.0.1:1234`
- Goal: check whether the pair is promising for speculative-style use and whether LM Studio overlaps both models when both are already loaded

## Results

### Chunk agreement

Source: `artifacts/lfm_pair_custom_ssd.json`

- Draft mean throughput: `105.46 tok/s`
- Target mean throughput: `32.35 tok/s`
- Mean acceptance ratio: `0.000`

Across all three 64-token iterations, accepted prefix length was `0`. The two models answered the same prompt in materially different ways from the first generated token onward.

### Sequential vs concurrent pair execution

Source: `artifacts/lfm_pair_concurrency.json`

| Mode | Mean wall s | Draft tok/s | Target tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 2.63 | 110.69 | 31.22 | 0.000 |
| Concurrent | 2.33 | 69.39 | 27.55 | 0.000 |

Concurrent wall-clock time improved by `1.13x` versus running the requests one after the other, which means LM Studio is overlapping work for the loaded pair. But per-model throughput dropped under overlap, especially for the 1.2B draft model, so the gain is limited.

## Conclusion

This pair is good for concurrency testing but bad for speculative decoding.

- Good news: both LFM models can run concurrently in LM Studio on this machine.
- Bad news: they are not aligned enough to use as a draft/target speculative pair in the current setup.
- Practical implication: use `lfm2.5-1.2b` as a fast small-model baseline and `lfm2-24b-a2b` as a large-model target, but do not expect acceptance-based speedups from pairing them directly.

## Files

- `reports/lfm_pair_custom_ssd.md`
- `reports/lfm_pair_concurrency.md`
- `artifacts/lfm_pair_custom_ssd.json`
- `artifacts/lfm_pair_concurrency.json`
