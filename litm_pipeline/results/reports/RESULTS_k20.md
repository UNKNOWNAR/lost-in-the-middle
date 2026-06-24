# Lost-in-the-Middle (LitM) Consolidated Evaluation Results (k=20)

This document summarizes the performance of **Qwen 2.5 7B Instruct** on the **119 Non-Factoid queries** across all 10 context placement conditions (k=20 total documents).

## Consolidated Metric Summary

Below is the consolidated performance table across all 10 conditions. The position of the 3 nugget-rich golden documents slides from the very beginning of the context window (Primacy) to the very end (Recency).

| Condition | Gold Doc Ranks | Vital Recall (%) | Okay Recall (%) | Support Score (%) | Hallucination Rate (%) | Mean Words | Median Words | Mode Words |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **k20_1** | 1-3 (Primacy) | **52.29%** | 30.67% | **89.17%** | 4.01% | 408.0 | 404 | 315 |
| **k20_2** | 3-5 | **54.44%** | 31.49% | **82.35%** | 9.49% | 412.1 | 417 | 321 |
| **k20_3** | 5-7 | **52.01%** | 31.08% | **82.15%** | 8.7% | 413.6 | 414 | 410 |
| **k20_4** | 7-9 | **51.58%** | 31.49% | **80.76%** | 10.35% | 410.0 | 413 | 369 |
| **k20_5** | 9-11 (Middle) | **52.29%** | 30.27% | **79.65%** | 11.01% | 407.5 | 415 | 354 |
| **k20_6** | 11-13 (Middle) | **53.15%** | 28.43% | **81.22%** | 9.75% | 409.0 | 403 | 376 |
| **k20_7** | 13-15 | **51.43%** | 28.63% | **79.21%** | 11.96% | 413.1 | 405 | 371 |
| **k20_8** | 15-17 | **50.29%** | 28.02% | **79.44%** | 12.08% | 407.7 | 399 | 370 |
| **k20_9** | 17-19 | **51.0%** | 27.81% | **80.77%** | 10.92% | 401.4 | 385 | 347 |
| **k20_10** | 18-20 (Recency) | **51.86%** | 31.29% | **79.91%** | 10.81% | 414.0 | 411 | 389 |

## Major Key Observations

1. **Lost-in-the-Middle Effect (U-Shape curve):**
   - Typically, Vital Recall is highest at the extreme ends of the prompt (Primacy and Recency) and lowest in the middle. We will examine if this pattern holds once the full run completes.
2. **Support Score vs Position:**
   - Evaluates if placing critical information in the middle causes the model to hallucinate more due to context neglect, or if it remains strictly faithful to the provided context.
3. **Word Count Consistency:**
   - Confirms that the model's verbosity remains stable across different prompt layouts, ensuring length bias does not skew the results.

---
> [!NOTE]
> All granular query-level evaluation logs are stored in:
> `litm_pipeline/data/processed/evaluations/`
