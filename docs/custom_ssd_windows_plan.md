# Custom SSD Plan For Windows

The upstream `tanishqkumar/ssd` runtime is a CUDA-first PyTorch engine. That is not the execution surface available on this machine.

What is available locally is different:

- LM Studio has installed `llama.cpp` runtimes for:
  - CPU
  - AMD ROCm
  - Vulkan
- LM Studio server logs on March 5, 2026 show real GPU use on this host:
  - `load_tensors: offloaded ... layers to GPU`
  - `clip_ctx: CLIP using Vulkan0 backend`

That means the practical Windows path is:

1. Keep the upstream SSD repo as reference architecture.
2. Build a local SSD-style prototype on top of LM Studio / `llama.cpp`.
3. Benchmark draft speed, target speed, agreement, and concurrency using the GPU-backed runtime that already works here.

## Current Prototype

The repo now contains a minimal custom scaffold:

- `custom_ssd/lmstudio_api.py`
  - Thin LM Studio OpenAI-compatible client.
- `custom_ssd/speculative.py`
  - Chunked draft/target agreement benchmark.
- `scripts/benchmark_custom_ssd.py`
  - CLI entry point that writes JSON and Markdown reports.

This is not a full reproduction of the research runtime. The LM Studio API does not expose the low-level verifier hooks, KV-cache control, or token-level acceptance machinery used by the paper implementation.

What it does give us:

- a Windows-native GPU-backed runtime,
- automated draft vs target comparison,
- measured chunk agreement,
- a basis for selecting viable draft/target pairs before attempting a deeper backend integration.

## Example Run

```powershell
python .\scripts\benchmark_custom_ssd.py `
  --draft-model google/gemma-3-1b `
  --target-model google/gemma-3-4b `
  --chunk-tokens 32 `
  --iterations 3
```

## Next Engineering Step

If this prototype shows good agreement for a draft/target pair, the next serious step is a tighter integration against a local `llama.cpp` runtime surface instead of the HTTP API, because that is where token-level acceptance and lower overhead become realistic.
