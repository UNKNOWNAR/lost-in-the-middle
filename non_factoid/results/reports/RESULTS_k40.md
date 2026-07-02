# Lost-in-the-Middle (LitM) Consolidated Evaluation Results (k=40)

This document summarizes the performance of **Qwen 2.5 7B Instruct** on the **119 Non-Factoid queries** across all 10 context placement conditions (k=40 total documents).

## Consolidated Metric Summary

Below is the consolidated performance table across all 10 conditions. The position of the 3 nugget-rich golden documents slides from the very beginning of the context window (Primacy) to the very end (Recency).

| Condition | Gold Doc Ranks | Vital Recall (%) | Okay Recall (%) | Support Score (%) | Hallucination Rate (%) | Mean Words | Median Words | Mode Words |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **k40_1** | 1-3 (Primacy) | **58.45%** | 32.92% | **85.01%** | 9.24% | 438.2 | 435 | 376 |
| **k40_2** | 5-7 | **56.88%** | 29.04% | **85.42%** | 8.7% | 430.5 | 434 | 503 |
| **k40_3** | 9-11 | **57.02%** | 23.52% | **84.27%** | 10.26% | 440.5 | 430 | 470 |
| **k40_4** | 13-15 | **56.02%** | 28.63% | **83.32%** | 10.95% | 436.6 | 440 | 362 |
| **k40_5** | 18-20 (Middle) | **54.58%** | 28.02% | **84.3%** | 9.95% | 438.4 | 431 | 232 |
| **k40_6** | 22-24 (Middle) | **54.01%** | 26.38% | **84.2%** | 10.37% | 436.4 | 430 | 437 |
| **k40_7** | 26-28 | **55.01%** | 28.02% | **84.77%** | 10.36% | 442.6 | 445 | 494 |
| **k40_8** | 30-32 | **55.01%** | 26.79% | **83.18%** | 11.58% | 443.7 | 450 | 450 |
| **k40_9** | 34-36 | **55.87%** | 28.63% | **83.94%** | 10.3% | 443.7 | 439 | 332 |
| **k40_10** | 38-40 (Recency) | **55.59%** | 29.45% | **83.32%** | 11.82% | 444.7 | 446 | 414 |

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
