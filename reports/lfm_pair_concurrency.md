# LM Studio Pair Concurrency Benchmark

- Draft model: `liquid/lfm2.5-1.2b`
- Target model: `liquid/lfm2-24b-a2b`
- Prompt: `Explain how multi-head attention, KV head sharing, expert routing, and concurrent execution affect local inference throughput on a workstation. Keep it technical.`
- Iterations: `3`
- Warmup rounds: `1`

| Mode | Mean wall s | Draft tok/s | Target tok/s | Acceptance | Concurrent gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential | 2.63 | 110.69 | 31.22 | 0.000 |  |
| Concurrent | 2.33 | 69.39 | 27.55 | 0.000 | 1.13x |

## Iterations

| Mode | Iter | Wall s | Draft s | Target s | Draft tok/s | Target tok/s | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential | 0 | 2.54 | 0.56 | 1.97 | 113.56 | 32.46 | 0.000 |
| sequential | 1 | 2.63 | 0.59 | 2.04 | 108.09 | 31.43 | 0.000 |
| sequential | 2 | 2.73 | 0.58 | 2.15 | 110.44 | 29.77 | 0.000 |
| concurrent | 0 | 2.23 | 0.90 | 2.22 | 70.89 | 28.78 | 0.000 |
| concurrent | 1 | 2.37 | 0.93 | 2.37 | 69.10 | 26.97 | 0.000 |
| concurrent | 2 | 2.38 | 0.94 | 2.38 | 68.17 | 26.91 | 0.000 |
