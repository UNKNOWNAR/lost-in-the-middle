# Lost-in-the-Middle (LitM) Consolidated Evaluation Results (k=60)

This document tracks the live evaluation progress and performance of **Qwen 2.5 7B Instruct** on the **102 Valid Non-Factoid queries** across all 10 context placement conditions (k=60 total documents).

*(Note: 17 degenerate queries have been excluded from this set. Safety filter blocks are skipped to be retried.)*

## Consolidated Metric Summary

| Condition | Gold Doc Ranks | Vital Recall (%) | Okay Recall (%) | Progress |
|:---:|:---:|:---:|:---:|:---:|
| **k60_1** | 1-3 (Primacy) | **51.45%** | 26.35% | **100/100 (100.0%) ✅** |
| **k60_2** | 7-9 | **49.05%** | 21.22% | **100/100 (100.0%) ✅** |
| **k60_3** | 14-16 | **44.07%** | 18.95% | **100/100 (100.0%) ✅** |
| **k60_4** | 21-23 | **42.40%** | 18.62% | **100/100 (100.0%) ✅** |
| **k60_5** | 28-30 (Middle) | **43.18%** | 18.92% | **100/100 (100.0%) ✅** |
| **k60_6** | 35-37 | **45.46%** | 20.35% | **100/100 (100.0%) ✅** |
| **k60_7** | 42-44 | **43.00%** | 20.74% | **100/100 (100.0%) ✅** |
| **k60_8** | 48-50 | **43.63%** | 19.47% | **100/100 (100.0%) ✅** |
| **k60_9** | 54-56 | **47.72%** | 21.41% | **100/100 (100.0%) ✅** |
| **k60_10** | 58-60 (Recency) | **47.52%** | 23.96% | **100/100 (100.0%) ✅** |
