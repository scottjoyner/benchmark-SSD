# LM Studio Full Matrix - 2026-03-05

Cold rows are first-request timings after model switch. Warm rows are a second single request against the same model. Parallel rows use 4 simultaneous requests.

| Model | Kind | Params | Heads | KV Heads | Layers | Context | Quant | Size GiB | Cold s | Warm tok/s | Parallel agg tok/s | Parallel req/s |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| liquid/lfm2-24b-a2b | moe | 64x1.3B | 32 | [0, 8] | 40 | 128000 | Q4_K_M | 13.43 | 17.82 | 23.96 | 44.72 | 1.40 |
| qwen/qwen3.5-35b-a3b | moe | 35B-A3B | 16 | 2 | 40 | 262144 | Q4_K_M | 19.72 | 55.68 | 8.68 | 5.67 | 0.18 |
| text-embedding-nomic-embed-text-v1.5 | embedding |  | 12 | None | 12 | 2048 | Q4_K_M | 0.08 | 1.73 |  |  | 94.35 |
| glm-4.7-flash | moe | 64x2.6B | 20 | 1 | 47 | 202752 | Q4_K_S | 16.08 | 36.21 | 26.09 | 34.14 | 1.07 |
| zai-org/glm-4.6v-flash | vision | 9.4B | 32 | 2 | 40 | 131072 | Q4_K_M | 5.74 | 13.62 | 31.55 | 60.88 | 1.90 |
| devstral-small-2-24b-instruct-2512 | text | 24B | 32 | 8 | 40 | 393216 | Q4_K_M | 13.35 | 28.95 | 6.76 | 11.15 | 0.35 |
| mistralai/ministral-3-3b | text | 3B | 32 | 8 | 26 | 262144 | Q4_K_M | 2.00 | 7.69 | 60.26 | 64.00 | 2.00 |
| trinity-mini | moe |  | 32 | 4 | 32 | 131072 | Q3_K_S | 10.80 | 14.78 | 37.50 | 53.90 | 1.68 |
| microsoft/phi-4-reasoning-plus | text | 15B | 40 | 10 | 40 | 32768 | Q4_K_M | 8.43 | 13.86 | 21.87 | 17.44 | 0.55 |
| google/gemma-3-4b | text | 4B | 8 | 4 | 34 | 131072 | Q4_K_M | 2.32 | 9.60 | 55.35 | 113.47 | 3.55 |
| google/gemma-3-27b | text | 27B | 32 | 16 | 62 | 131072 | Q4_0 | 14.50 | 42.21 | 4.90 | 8.38 | 0.26 |
| deepseek/deepseek-r1-0528-qwen3-8b | text | 8B | 32 | 8 | 36 | 131072 | Q4_K_M | 4.68 | 8.37 | 38.61 | 79.58 | 2.49 |
| mistralai/mistral-7b-instruct-v0.3 | text | 7B | 32 | 8 | 32 | 32768 | Q4_K_M | 4.07 | 5.06 | 45.56 | 90.56 | 2.83 |
| qwen/qwen3-vl-4b | vision | 4B | 32 | 8 | 36 | 262144 | Q4_K_M | 2.33 | 4.28 | 66.07 | 134.26 | 4.20 |
| microsoft/phi-4-mini-reasoning | text | 3B | 24 | 8 | 32 | 131072 | Q4_K_M | 2.32 | 3.60 | 67.53 | 138.29 | 4.32 |
| text-embedding-all-minilm-l6-v2 | embedding | 22M | 12 | None | 6 | 512 | Q8_0 | 0.02 | 0.67 |  |  | 188.73 |
| text-embedding-all-minilm-l6-v2-embedding | embedding | 22M | 12 | None | 6 | 512 | Q8_0 | 0.02 | 1.43 |  |  | 133.98 |
| text-embedding-all-minilm-l12-v2 | embedding | 33M | 12 | None | 12 | 512 | Q8_0 | 0.03 | 0.63 |  |  | 131.03 |
| google/gemma-3-1b | text | 1B | 4 | 1 | 26 | 32768 | Q4_0 | 0.67 | 1.75 | 136.38 | 292.45 | 9.14 |
| ibm/granite-4-h-tiny | moe | 7B | 12 | [0, 4] | 40 | 1048576 | Q4_K_M | 3.94 | 4.09 | 75.73 | 140.48 | 4.39 |
| openai/gpt-oss-20b | moe | 20B | 64 | 8 | 24 | 131072 | MXFP4 | 11.28 | 16.34 | 28.52 | 40.18 | 1.26 |
