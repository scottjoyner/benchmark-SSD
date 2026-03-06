# LM Studio Pair Concurrency Benchmark

- Draft model: `google/gemma-3-4b`
- Target model: `google/gemma-3-27b`
- Prompt: `Explain how multi-head attention, KV head sharing, expert routing, and concurrent execution affect local inference throughput on a workstation. Keep it technical.`
- Iterations: `3`
- Warmup rounds: `1`

| Mode | Mean wall s | Draft tok/s | Target tok/s | Acceptance | Concurrent gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential | 16.28 | 19.84 | 4.90 | 0.000 |  |
| Concurrent | 14.34 | 16.89 | 4.47 | 0.000 | 1.14x |

## Iterations

| Mode | Iter | Wall s | Draft s | Target s | Draft tok/s | Target tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential | 0 | 16.35 | 3.19 | 13.15 | 20.04 | 4.87 | 0.000 |
| sequential | 1 | 16.18 | 3.32 | 12.86 | 19.26 | 4.98 | 0.000 |
| sequential | 2 | 16.31 | 3.16 | 13.15 | 20.22 | 4.87 | 0.000 |
| concurrent | 0 | 14.46 | 3.77 | 14.43 | 16.99 | 4.44 | 0.000 |
| concurrent | 1 | 14.20 | 3.79 | 14.19 | 16.89 | 4.51 | 0.000 |
| concurrent | 2 | 14.37 | 3.82 | 14.36 | 16.78 | 4.46 | 0.000 |
