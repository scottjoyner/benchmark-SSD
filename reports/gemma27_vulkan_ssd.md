# Standalone Vulkan SSD Benchmark

- Draft model: `C:\Users\AMD\.lmstudio\models\lmstudio-community\gemma-3-4b-it-GGUF\gemma-3-4b-it-Q4_K_M.gguf`
- Target model: `C:\Users\AMD\.lmstudio\models\lmstudio-community\gemma-3-27B-it-qat-GGUF\gemma-3-27B-it-QAT-Q4_0.gguf`

| Algorithm | Output tok | Tok/s | Prompt s | Gen s | Acceptance | Branch hit | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autoregressive | 16 | 3.64 | 9.91 | 4.39 |  |  |  |
| sync_speculative | 16 | 1.36 | 9.02 | 11.79 | 0.500 |  | 6 |
| async_prefetch | 16 | 1.14 | 8.00 | 14.02 | 0.500 | 0.333 | 6 |
