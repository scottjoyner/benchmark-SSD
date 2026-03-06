# Benchmark Report - 2026-03-05

## Scope

This report compares:

- Measured local baseline inference through LM Studio's OpenAI-compatible server on this Windows 11 Ryzen AI system.
- The runtime design and execution requirements of the upstream `ssd` package cloned into `upstream-ssd/`.

`ssd` itself was not executed on this machine because the upstream package currently depends on CUDA/Triton/FlashInfer/NCCL-oriented infrastructure that is not available in this Windows AMD setup.

## Local baseline runs

Prompt set:

- `prompts/benchmark_prompts.jsonl`
- 4 prompts
- `max_tokens=128`
- `temperature=0.0`

Reports:

- `artifacts/lmstudio-benchmarks/google__gemma-3-1b.json`
- `artifacts/lmstudio-benchmarks/google__gemma-3-4b.json`
- `artifacts/lmstudio-benchmarks/mistralai__mistral-7b-instruct-v0.3.json`
- `artifacts/lmstudio-benchmarks/deepseek__deepseek-r1-0528-qwen3-8b.json`
- `artifacts/lmstudio-benchmarks/openai__gpt-oss-20b.json`

## Results

Cold numbers include model load and warmup on the first request after switching models. Warm numbers are the mean of prompts 2 through 4 and are the better steady-state inference estimate.

| Model | Cold elapsed | Cold tok/s | Warm mean elapsed | Warm mean tok/s |
| --- | ---: | ---: | ---: | ---: |
| `google/gemma-3-1b` | 10.72 s | 11.94 | 1.70 s | 75.70 |
| `google/gemma-3-4b` | 18.46 s | 6.93 | 3.51 s | 36.66 |
| `openai/gpt-oss-20b` | 62.81 s | 2.04 | 5.05 s | 25.40 |
| `mistralai/mistral-7b-instruct-v0.3` | 14.69 s | 8.71 | 6.12 s | 20.91 |
| `deepseek/deepseek-r1-0528-qwen3-8b` | 18.91 s | 6.77 | 6.76 s | 18.95 |

## Observations

1. The local baseline is strongly bifurcated into cold-start cost and steady-state decode speed.
2. `gemma-3-1b` is the throughput leader by a large margin at about `75.7 tok/s` steady-state.
3. `gemma-3-4b` approximately halves throughput versus `1b`, which is a reasonable scaling trend for this host.
4. `mistral-7b` and `deepseek-r1-qwen3-8b` land in a similar steady-state band around `19-21 tok/s`.
5. `gpt-oss-20b` has the heaviest cold-start penalty at `62.81 s`, but its steady-state speed of `25.4 tok/s` is still competitive with the 7B and 8B models. That indicates backend implementation details and architecture matter here, not only parameter count.

## SSD runtime analysis

### Package-level constraints

From `upstream-ssd/pyproject.toml`:

- `torch==2.8.0`
- `triton==3.4.0`
- `flashinfer-python==0.5.2`
- `sgl-kernel==0.3.17.post1`
- `nvidia-cutlass-dsl==4.2.1`

This is a CUDA-native runtime stack, not a portable Python package with a generic CPU path.

### Engine shape

From `upstream-ssd/ssd/engine/llm_engine.py`:

- SSD launches multiple worker processes with `torch.multiprocessing`.
- Async draft mode requires more than one GPU.
- In async mode, the target model and draft model are split across distinct devices.

From `upstream-ssd/ssd/engine/model_runner.py`:

- Execution assumes `torch.cuda.set_device(...)`.
- Distributed initialization uses NCCL over localhost.
- FlashInfer wrappers and CUDA graph capture are part of normal setup, not optional side paths.

From `upstream-ssd/ssd/engine/speculator_async.py` and `upstream-ssd/ssd/engine/draft_runner.py`:

- The async draft worker and target worker exchange speculation metadata and logits over a process group.
- SSD preallocates GPU-side handshake, cache, and logit buffers to keep speculative steps cheap.
- Tree-cache hits are fundamental to the async design: the draft side tries to answer likely target outcomes before verification completes.

From `upstream-ssd/ssd/engine/verifier.py`:

- Verification always runs on the target model and measures accepted suffix lengths, verify time, cache-hit behavior, and effective tokens accepted per step.
- The main runtime win comes from collapsing draft overhead when the target's verification outcome has already been anticipated and cached.

## Comparison against the local baseline

The LM Studio baseline you ran here is conventional local inference with one active model endpoint at a time. Even when it uses an optimized backend internally, its runtime model is still basically:

1. Load one model.
2. Run a standard prompt/completion request.
3. Return tokens after the backend finishes normal decode steps.

The upstream SSD runtime is materially different:

1. It requires two inference roles, target and draft.
2. In async mode those roles live on separate GPUs.
3. The draft side speculates across multiple possible verification branches.
4. Verification either converts that speculation into an immediate accepted suffix or falls back to recovery.

So the comparison is not just "one package is faster than another package." It is:

- Baseline runtime: single active inference path.
- SSD runtime: multi-process, multi-GPU, speculative tree execution with explicit inter-process coordination.

## Practical conclusion for this machine

For this Windows Ryzen AI host, the practical benchmark result today is the LM Studio baseline above. The upstream SSD package is not in a runnable state here without a substantial port or a move to Linux plus compatible NVIDIA GPUs.

If the goal is a faithful paper-style SSD comparison, the next valid step is:

1. Reuse this prompt set on a Linux + NVIDIA machine.
2. Run `upstream-ssd/bench/bench.py` there.
3. Compare SSD's accepted-suffix and throughput metrics against the local LM Studio baseline captured in `artifacts/lmstudio-benchmarks/`.
