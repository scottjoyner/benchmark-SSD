# SSD on This Windows Host

## Summary

The upstream `ssd` repository is present in `upstream-ssd/`, but it is not ready to run on this Windows 11 Ryzen AI system without major adaptation.

## Why it does not run here yet

- The upstream README states `Python 3.11+` and `CUDA >= 12.8`.
- The upstream `pyproject.toml` pins `torch==2.8.0`, `triton==3.4.0`, `flashinfer-python==0.5.2`, `sgl-kernel==0.3.17.post1`, and `nvidia-cutlass-dsl==4.2.1`.
- The benchmark examples assume multi-GPU NVIDIA layouts such as H100, A100, L40, or 4090-class CUDA environments.
- The code imports CUDA-specific runtime pieces early and uses Triton/FlashInfer kernels, so this is not a simple CPU fallback scenario.

## What is still useful now

- The repo is available locally for algorithm study and later porting.
- The prompt set and baseline measurement scripts in this workspace can be reused for apples-to-apples comparisons.
- LM Studio already has bundled `llama.cpp` backends and a populated GGUF model cache, so a strong conventional baseline can be measured on this device immediately.

## Realistic next options

1. Run the paper's SSD implementation on separate Linux + NVIDIA hardware and compare against the same prompt set captured here.
2. Investigate a port to a different backend, understanding that this is research-engineering work rather than setup work.
3. Use this machine only for the baseline side of the comparison and store results under `artifacts/` for later merging.
