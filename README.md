# benchmark-SSD

Workspace for comparing the upstream `ssd` research codebase with a practical local baseline on this Windows 11 Ryzen AI machine.

## What is in this repo

- `upstream-ssd/`: cloned copy of https://github.com/tanishqkumar/ssd for inspection and later Linux/NVIDIA runs.
- `scripts/discover_lmstudio_models.py`: scans `C:\Users\AMD\.lmstudio\models` and ranks benchmark candidates.
- `scripts/benchmark_lmstudio_openai.py`: benchmarks a loaded LM Studio model through its OpenAI-compatible API.
- `scripts/setup_workspace.ps1`: creates local artifact folders and snapshots the current model inventory.
- `prompts/benchmark_prompts.jsonl`: starter prompts for throughput and latency checks.
- `docs/ssd_windows_status.md`: notes on what blocks native SSD execution on this host.

## Current status

This machine already has LM Studio models under `C:\Users\AMD\.lmstudio\models` and LM Studio is configured for an OpenAI-compatible HTTP server on port `1234`, but the server was not reachable during setup, so it likely needs to be started manually before baseline benchmarks run.

The upstream SSD code is not directly runnable on this host as-is. As of March 5, 2026 the upstream repository pins CUDA- and Triton-specific dependencies (`torch==2.8.0`, `triton==3.4.0`, `flashinfer-python`, `sgl-kernel`) and documents `CUDA >= 12.8` with testing on H100-class NVIDIA systems.

## Recommended local benchmark models

For this 32 GB Windows host, start with:

- `lmstudio-community/gemma-3-1B-it-QAT-GGUF/gemma-3-1B-it-QAT-Q4_0.gguf`
- `lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf`
- `lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
- `lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-GGUF/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf`

These are small enough to be realistic baselines while still covering a useful quality/performance range.

## Quick start

Create the workspace artifacts and snapshot the current model inventory:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_workspace.ps1
```

Review the model inventory:

```powershell
python .\scripts\discover_lmstudio_models.py
```

Once LM Studio's local server is running and a model is loaded, benchmark it:

```powershell
python .\scripts\benchmark_lmstudio_openai.py `
  --model lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf `
  --prompt-file .\prompts\benchmark_prompts.jsonl `
  --samples 4 `
  --max-tokens 128
```

Results are written to `artifacts\lmstudio-benchmarks\`.

## Near-term plan

1. Establish reproducible local baseline numbers with LM Studio / llama.cpp using the existing GGUF cache.
2. Decide whether SSD evaluation should happen on separate compatible hardware or via a porting effort.
3. If compatible hardware is available, mirror the same prompts and metrics against `upstream-ssd/bench/bench.py`.
