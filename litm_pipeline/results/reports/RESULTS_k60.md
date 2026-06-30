# Lost-in-the-Middle (LitM) Consolidated Evaluation Results (k=60)

This document tracks the live evaluation progress and performance of **Qwen 2.5 7B Instruct** on the **102 Valid Non-Factoid queries** across all 10 context placement conditions (k=60 total documents).

*(Note: 17 degenerate queries have been excluded from this set. Safety filter blocks are skipped to be retried.)*

## Consolidated Metric Summary

| Condition | Gold Doc Ranks | Vital Recall (%) | Okay Recall (%) | Progress |
|:---:|:---:|:---:|:---:|:---:|
| **k60_1** | 1-3 (Primacy) | **51.65%** | 26.75% | 101/102 (99.0%) |
| **k60_2** | 7-9 | **47.95%** | 20.29% | 98/102 (96.1%) |
| **k60_3** | 14-16 | **44.26%** | 18.66% | 98/102 (96.1%) |
| **k60_4** | 21-23 | **42.43%** | 18.32% | 98/102 (96.1%) |
| **k60_5** | 28-30 (Middle) | **47.21%** | 24.80% | 45/102 (44.1%) |
| **k60_6** | 35-37 | **N/A** | N/A | 0/102 (0.0%) |
| **k60_7** | 42-44 | **N/A** | N/A | 0/102 (0.0%) |
| **k60_8** | 48-50 | **N/A** | N/A | 0/102 (0.0%) |
| **k60_9** | 54-56 | **N/A** | N/A | 0/102 (0.0%) |
| **k60_10** | 58-60 (Recency) | **N/A** | N/A | 0/102 (0.0%) |
