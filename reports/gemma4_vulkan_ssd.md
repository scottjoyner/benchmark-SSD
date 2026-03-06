# Standalone Vulkan SSD Benchmark

- Draft model: `C:\Users\AMD\.lmstudio\models\lmstudio-community\gemma-3-1B-it-QAT-GGUF\gemma-3-1B-it-QAT-Q4_0.gguf`
- Target model: `C:\Users\AMD\.lmstudio\models\lmstudio-community\gemma-3-4b-it-GGUF\gemma-3-4b-it-Q4_K_M.gguf`

| Algorithm | Output tok | Tok/s | Prompt s | Gen s | Acceptance | Branch hit | Rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autoregressive | 16 | 64.92 | 2.70 | 0.25 |  |  |  |
| sync_speculative | 16 | 5.96 | 0.23 | 2.68 | 0.393 |  | 7 |
| async_prefetch | 16 | 5.21 | 0.38 | 3.07 | 0.393 | 0.143 | 7 |
