# Gemma Focus Update

## Scope

- Focused only on the Gemma family
- Runtime: standalone Vulkan exact speculative path
- Models:
  - `gemma-3-1B -> gemma-3-4B`
  - `gemma-3-4B -> gemma-3-27B`
- Sweep: `k = 2, 4, 6`
- Important runtime condition: the Gemma models were also live in LM Studio during this sweep

## Results

Source: `artifacts/gemma_vulkan_sweep.json`

| Pair | k | AR tok/s | Sync tok/s | Async tok/s | Sync accept | Async accept | Branch hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 1B -> 4B | 2 | 48.59 | 2.34 | 3.37 | 0.500 | 0.500 | 0.375 |
| Gemma 1B -> 4B | 4 | 46.77 | 5.31 | 4.91 | 0.393 | 0.393 | 0.143 |
| Gemma 1B -> 4B | 6 | 47.03 | 6.04 | 5.16 | 0.389 | 0.389 | 0.000 |
| Gemma 4B -> 27B | 2 | 0.26 | 0.17 | 0.22 | 0.786 | 0.786 | 0.429 |
| Gemma 4B -> 27B | 4 | 0.31 | 0.28 | 0.21 | 0.500 | 0.500 | 0.333 |
| Gemma 4B -> 27B | 6 | 0.33 | 0.35 | 0.33 | 0.500 | 0.500 | 0.250 |

## Interpretation

For `Gemma 1B -> 4B`, larger `k` helped the exact path a bit, but it still stayed far below plain AR. The best result in this sweep was sync speculative at `k=6`, reaching `6.04 tok/s` against a `47.03 tok/s` AR baseline.

For `Gemma 4B -> 27B`, `k=6` was the first point that roughly matched or slightly edged AR in this run: `0.35 tok/s` sync versus `0.33 tok/s` AR. That is the most interesting Gemma-only result so far.

But the absolute `4B -> 27B` numbers were much slower than the earlier clean standalone run. The likely reason is resource contention: this sweep was run while LM Studio also had the Gemma endpoint models loaded. So these numbers are best interpreted as **shared-runtime coexistence results**, not clean standalone maxima.

## Practical Takeaway

- Stay on Gemma for exact speculative work.
- If the goal is algorithm evaluation, unload the Gemma models from LM Studio before rerunning the standalone Vulkan path.
- If the goal is coexistence testing, this sweep already shows the shared-runtime penalty clearly.
- Among the tested values, `k=6` is the best next default for Gemma.

## Files

- `reports/gemma_vulkan_sweep.md`
- `artifacts/gemma_vulkan_sweep.json`
- `reports/vulkan_standalone_analysis_2026-03-06.md`
