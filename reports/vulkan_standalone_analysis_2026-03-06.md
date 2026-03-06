# Standalone Vulkan SSD Analysis

## Setup

- Runtime: standalone `llama.cpp` Vulkan backend from LM Studio's local backend pack
- Host: Windows 11, Ryzen AI Max Pro 390 / Radeon 8050S-class Vulkan device
- Prompt: `Explain how speculative decoding, KV cache reuse, attention head layout, and CPU/GPU overlap affect local LLM inference throughput on a workstation.`
- Decode config: `max_new_tokens=16`, `n_ctx=1024`, `n_batch=128`, greedy decode
- SSD implementation: exact draft/verify path in `custom_ssd/standalone_speculative.py`

## Autoregressive Matrix

| Model | Arch | Quant | Layers | Heads | KV heads | Experts / active | GPU layers | Tok/s | Prompt s | Gen s |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Gemma 3 4B | gemma3 | Q4_K_M | 34 | 8 | 4 | dense | `-1` | 64.92 | 2.70 | 0.25 |
| GPT-OSS 20B | gpt-oss | MXFP4 | 24 | 64 | 8 | 32 / 4 | 24 | 27.44 | 3.89 | 0.58 |
| LFM2 24B A2B | lfm2moe | Q4_K_M | 40 | 32 | mixed | 64 / 4 | 24 | 28.83 | 6.83 | 0.55 |
| Qwen3.5 35B A3B | qwen35moe | Q4_K_M | 40 | 16 | 2 | 256 / 8 | 24 | 8.19 | 9.03 | 1.95 |
| Devstral Small 2 24B | mistral3 | Q4_K_M | 40 | 32 | 8 | dense | 24 | 6.34 | 5.11 | 2.52 |
| Gemma 3 27B | gemma3 | Q4_0 | 62 | 32 | 16 | dense | 24 | 3.64 | 9.91 | 4.39 |

Raw artifacts:

- `artifacts/gptoss20b_vulkan_ar_24layers.json`
- `artifacts/lfm2_24b_vulkan_ar_24layers.json`
- `artifacts/qwen35_a3b_vulkan_ar_24layers.json`
- `artifacts/devstral24b_vulkan_ar_24layers.json`
- `artifacts/gemma27_vulkan_ssd.json`
- `artifacts/gemma4_vulkan_ssd.json`

## Exact Speculative Matrix

Tokenizer-compatible exact pairs found locally:

- `gemma-3-1B -> gemma-3-4B`
- `gemma-3-4B -> gemma-3-27B`

Pairs rejected for exact mode due tokenizer mismatch:

- `Ministral-3-3B -> Mistral-7B`
- `Phi-4-mini -> Phi-4-reasoning-plus`
- `DeepSeek-R1-Qwen3-8B -> Qwen3.5-35B-A3B`

| Draft -> Target | Target GPU layers | Target AR tok/s | Sync tok/s | Async tok/s | Acceptance | Branch hit | Sync vs AR | Async vs AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 1B -> Gemma 4B | `-1` | 64.92 | 5.96 | 5.21 | 0.393 | 0.143 | 0.09x | 0.08x |
| Gemma 4B -> Gemma 27B | 24 | 3.64 | 1.36 | 1.14 | 0.500 | 0.333 | 0.37x | 0.31x |

Raw artifacts:

- `artifacts/gemma4_vulkan_ssd.json`
- `artifacts/gemma27_vulkan_ssd.json`

## Analysis

The standalone Vulkan path is real and substantially faster than the earlier untuned runs once GPU layer count is tuned for the local UMA budget. `gpt-oss-20b` moved from an unusable cold-path result to `27.44 tok/s` with a 24-layer Vulkan split, and `LFM2-24B-A2B` landed at `28.83 tok/s`. That is the main practical result from this round: the AMD Vulkan backend is viable for large-model standalone inference on this machine.

Attention heads alone do not predict throughput. `GPT-OSS 20B` has `64` attention heads and still outperformed dense `Gemma 27B` by a wide margin, while `Qwen3.5-35B-A3B` with only `16` heads was much slower than both `GPT-OSS` and `LFM2`. The stronger predictors here were active-parameter footprint, quantization format, and how much of the model could be kept on the Vulkan path. `MXFP4` on `GPT-OSS` and the lower active-expert count on the MoE models both mattered more than raw head count.

Parallel draft/verify on one Windows AMD system did not beat single-model AR in the current exact implementation. That is not a contradiction of the SSD paper. SSD's advantage depends on true hardware separation and low-overhead branch management. This prototype is exact, but it pays a large cost for llama.cpp context snapshot and restore on every verify round. On this host, async overlap also competes for the same Vulkan device and memory system, so the overlap path increased contention instead of removing it.

The Gemma 27B run is the clearest result for "parallel vs single execution" on this machine:

- Single target AR: `3.64 tok/s`
- Exact sync speculative: `1.36 tok/s`
- Exact async prefetch: `1.14 tok/s`

So the standalone engine is working, but the present bottleneck is implementation overhead, not model quality. To move this closer to SSD-style speedups on Windows/Vulkan, the next work should target cheaper branch state management and deeper backend integration rather than more prompt-level orchestration.

## Practical Conclusions

- Use standalone Vulkan AR for the larger local models now. `GPT-OSS 20B` and `LFM2-24B-A2B` are the best large-model performers measured in this run.
- Treat exact speculative mode as a research path on this machine, not a production speed path yet.
- Restrict exact speculative experiments to tokenizer-identical model families. On the local inventory, that effectively means Gemma right now.
- When comparing architectures, prioritize active experts, quantization, and fit/offload depth over raw attention-head count.
