# Final Analysis - 2026-03-05

## Scope

This run covered:

- All models exposed by LM Studio's local API on March 5, 2026.
- Single-request cold and warm behavior.
- Four-way parallel request behavior.
- Local architecture metadata from LM Studio caches, including attention heads, KV heads, layers, context length, quantization, and file size.
- A direct SSD runtime compatibility check on this Windows host.

Benchmark harness:

- Prompt type: single fixed technical text prompt for generative models.
- Output length: `32` completion tokens.
- Embedding models: single fixed text input.
- Parallel test: `4` simultaneous requests per model.

Primary artifacts:

- `artifacts/full_lmstudio_sweep.json`
- `reports/lmstudio_full_matrix_2026-03-05.md`

## SSD status on this host

The upstream SSD repository is available in `upstream-ssd/`, but direct SSD benchmarks were not runnable here.

Why:

1. The upstream package pins CUDA-native dependencies such as `triton`, `flashinfer-python`, `sgl-kernel`, and `nvidia-cutlass-dsl`.
2. The runtime code assumes CUDA device selection, NCCL process groups, and multi-GPU async target/draft execution.
3. The available local Python interpreters are `3.14` and `3.13`, while upstream SSD declares `>=3.11,<3.13`.
4. A local package-install attempt under Python `3.14` failed before dependency resolution with a permissions error in the user temp/pip build path, and a Python `3.13` venv creation attempt also failed with access denied. Even without those permission issues, the Python-version and CUDA-runtime assumptions still make this host incompatible with the upstream SSD benchmark path.
5. An elevated WSL check showed a real `Ubuntu-24.04` install with Python `3.12.3`, but no visible NVIDIA stack. The WSL device surface only exposed `/dev/dxg` plus Microsoft virtual 3D adapters, and the environment had no `nvidia-smi` path. That is not a valid CUDA/NCCL execution surface for the upstream SSD runtime.

Conclusion:

- LM Studio / llama.cpp benchmarks below are real measured results on this machine.
- SSD results below are architectural and compatibility analysis, not measured SSD throughput numbers, because the upstream SSD engine is not runnable on this Windows AMD configuration as-is.

## Performance matrix

The full grid is in `reports/lmstudio_full_matrix_2026-03-05.md`. Key rows are repeated here for convenience.

| Model | Kind | Params | Heads | KV Heads | Size GiB | Cold s | Warm tok/s | Parallel agg tok/s |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `google/gemma-3-1b` | text | 1B | 4 | 1 | 0.67 | 1.75 | 136.38 | 292.45 |
| `ibm/granite-4-h-tiny` | moe | 7B | 12 | `[0,4]` pattern | 3.94 | 4.09 | 75.73 | 140.48 |
| `microsoft/phi-4-mini-reasoning` | text | 3B | 24 | 8 | 2.32 | 3.60 | 67.53 | 138.29 |
| `qwen/qwen3-vl-4b` | vision | 4B | 32 | 8 | 2.33 | 4.28 | 66.07 | 134.26 |
| `mistralai/ministral-3-3b` | text | 3B | 32 | 8 | 2.00 | 7.69 | 60.26 | 64.00 |
| `google/gemma-3-4b` | text | 4B | 8 | 4 | 2.32 | 9.60 | 55.35 | 113.47 |
| `mistralai/mistral-7b-instruct-v0.3` | text | 7B | 32 | 8 | 4.07 | 5.06 | 45.56 | 90.56 |
| `deepseek/deepseek-r1-0528-qwen3-8b` | text | 8B | 32 | 8 | 4.68 | 8.37 | 38.61 | 79.58 |
| `openai/gpt-oss-20b` | moe | 20B | 64 | 8 | 11.28 | 16.34 | 28.52 | 40.18 |
| `liquid/lfm2-24b-a2b` | moe | 64x1.3B | 32 | sparse `[0,8]` pattern | 13.43 | 17.82 | 23.96 | 44.72 |
| `glm-4.7-flash` | moe | 64x2.6B | 20 | 1 | 16.08 | 36.21 | 26.09 | 34.14 |
| `microsoft/phi-4-reasoning-plus` | text | 15B | 40 | 10 | 8.43 | 13.86 | 21.87 | 17.44 |
| `qwen/qwen3.5-35b-a3b` | moe | 35B-A3B | 16 | 2 | 19.72 | 55.68 | 8.68 | 5.67 |
| `google/gemma-3-27b` | text | 27B | 32 | 16 | 14.50 | 42.21 | 4.90 | 8.38 |

Embedding models:

| Model | Heads | Dims | Warm latency | Parallel req/s |
| --- | ---: | ---: | ---: | ---: |
| `text-embedding-nomic-embed-text-v1.5` | 12 | 768 | 0.0141 s | 94.35 |
| `text-embedding-all-minilm-l6-v2` | 12 | 384 | 0.0195 s | 188.73 |
| `text-embedding-all-minilm-l6-v2-embedding` | 12 | 384 | 0.0222 s | 133.98 |
| `text-embedding-all-minilm-l12-v2` | 12 | 384 | 0.0120 s | 131.03 |

## High-signal findings

### 1. Small dense models dominate single-request throughput

`google/gemma-3-1b` was the clear winner at `136.38 tok/s` warm and `292.45 tok/s` aggregated under four parallel requests.

That result matters because it shows this machine can drive very high local throughput when:

- model size is small,
- KV/cache demand is low,
- head count is modest,
- and cold-start weight loading is small.

### 2. Parallelism helps, but not equally across models

Some models nearly doubled or better under four-way parallel load:

- `google/gemma-3-1b`: `136.38 -> 292.45 tok/s`
- `google/gemma-3-4b`: `55.35 -> 113.47 tok/s`
- `microsoft/phi-4-mini-reasoning`: `67.53 -> 138.29 tok/s`
- `mistralai/mistral-7b-instruct-v0.3`: `45.56 -> 90.56 tok/s`

Others improved only slightly or even regressed:

- `mistralai/ministral-3-3b`: `60.26 -> 64.00 tok/s`
- `microsoft/phi-4-reasoning-plus`: `21.87 -> 17.44 tok/s`
- `qwen/qwen3.5-35b-a3b`: `8.68 -> 5.67 tok/s`

Interpretation:

- For smaller or well-optimized models, LM Studio/llama.cpp is able to exploit request-level concurrency effectively.
- For larger or more memory-bound models, four-way parallelism mostly adds contention instead of unlocking extra throughput.
- This supports your point about parallel vs single execution being a first-order variable on this machine.

### 3. Head count alone does not predict performance

The data does not support a simple "more attention heads = slower" rule.

Examples:

- `gemma-3-1b` has only `4` attention heads and is the fastest model tested.
- `granite-4-h-tiny` has `12` heads yet outperforms several 24-32 head dense models.
- `phi-4-mini-reasoning` has `24` heads and still performs extremely well.
- `gpt-oss-20b` has `64` heads and is much slower, but that slowdown is not explained by head count alone.

What appears to matter more in practice is the combination of:

- total active parameter footprint,
- quantization format,
- KV-head layout or grouped-query attention behavior,
- sequence/context defaults,
- and whether the model is compute-bound or memory-bound under concurrent load.

### 4. KV-head structure matters as much as head count

A recurring pattern in the faster models is relatively economical KV-head structure:

- `gemma-3-1b`: `4` heads / `1` KV head
- `gemma-3-4b`: `8` heads / `4` KV heads
- `phi-4-mini-reasoning`: `24` heads / `8` KV heads

By contrast, larger models with more expensive KV layouts and larger active parameter footprints tended to lose both cold-start and steady-state efficiency.

For the MoE models, KV structure can also be irregular or sparse by layer:

- `granite-4-h-tiny`: patterned KV distribution
- `lfm2-24b-a2b`: sparse `[0,8]`-style layer pattern

That makes simple apples-to-apples interpretation by "number of heads" misleading. The real bottleneck is the runtime's actual cache and routing behavior.

### 5. Cold-start cost becomes dominant for large models

Worst cold-starts in this run:

- `qwen/qwen3.5-35b-a3b`: `55.68 s`
- `google/gemma-3-27b`: `42.21 s`
- `glm-4.7-flash`: `36.21 s`
- `devstral-small-2-24b-instruct-2512`: `28.95 s`

These models can still produce usable steady-state speeds, but for interactive use the load/warmup penalty is material and can dominate end-user experience unless the model remains resident.

### 6. MoE models are mixed rather than universally better

MoE did not automatically win on this host.

Examples:

- Strong MoE result: `granite-4-h-tiny` at `75.73 tok/s`
- Middle-tier MoE: `gpt-oss-20b` at `28.52 tok/s`
- Weak MoE under this setup: `qwen3.5-35b-a3b` at `8.68 tok/s`

So on this machine, MoE quality/performance depends heavily on the specific model implementation and quantized artifact, not just the fact that it is sparse.

### 7. Vision-language models are viable text generators here

`qwen/qwen3-vl-4b` and `zai-org/glm-4.6v-flash` both performed well on plain text prompts:

- `qwen3-vl-4b`: `66.07 tok/s` warm
- `glm-4.6v-flash`: `31.55 tok/s` warm

That means the multimodal checkpoints are not merely "extra" models in the cache; they are legitimate local inference options for text workloads on this system.

### 8. Embedding models are effectively instant relative to generation models

Embedding latencies were measured in milliseconds, not seconds. The MiniLM embedding variants delivered very high request rates, which makes them suitable for retrieval and indexing workloads without materially competing with the heavier generative models for the same type of latency budget.

## Model-specific notes

- `devstral-small-2505` intermittently returned HTTP 400 to the benchmark harness even though a direct single API probe succeeded. That points to an API-format or prompt-handling edge case rather than a hard inability to run. It should be treated as "needs custom harness handling", not "unusable".
- `phi-4-reasoning-plus` had a notable concurrency regression, suggesting it is particularly sensitive to contention on this machine.
- `ministral-3-3b` had excellent warm single-request throughput but almost no parallel scaling, which suggests the backend is already close to saturating the useful execution path for that model.

## Practical recommendations

### Best pure speed choices

- `google/gemma-3-1b`
- `ibm/granite-4-h-tiny`
- `microsoft/phi-4-mini-reasoning`
- `qwen/qwen3-vl-4b`

### Best balanced local text models

- `google/gemma-3-4b`
- `mistralai/mistral-7b-instruct-v0.3`
- `deepseek/deepseek-r1-0528-qwen3-8b`

### Best choices when you want concurrency

- `google/gemma-3-1b`
- `microsoft/phi-4-mini-reasoning`
- `ibm/granite-4-h-tiny`
- `google/gemma-3-4b`

### Models to keep resident only if you really need them

- `qwen/qwen3.5-35b-a3b`
- `google/gemma-3-27b`
- `devstral-small-2-24b-instruct-2512`

These impose large cold-start penalties and do not reward this hardware with especially strong tokens/sec in return.
