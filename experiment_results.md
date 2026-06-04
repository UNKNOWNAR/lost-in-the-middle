# Lost in the Middle: Llama 3.1:8b Evaluation Report

This report summarizes the performance evaluation of the local `llama3.1:8b` model under the "Lost in the Middle" retrieval task. The evaluation covers three different context sizes (10, 20, and 30 total documents) across all 2,655 questions per position.

> **ℹ️ Note:** **Lost in the Middle Phenomenon:** Large Language Models (LLMs) are often much better at retrieving information from the absolute beginning or the absolute end of their input contexts, but their accuracy degrades significantly when relevant information is placed in the middle.

---

## 1. Detailed Performance Tables & Observations

### 10-Document Context Configuration
* **Total Run Time:** 4.2 Hours
* **Total Queries:** 7,965 (2,655 questions × 3 positions)

| Gold Position (Index) | Correct Predictions | Total Evaluated | Accuracy (%) |
| :---: | :---: | :---: | :---: |
| **0** (Beginning) | 1,314 | 2,655 | **49.49%** |
| **4** (Middle) | 1,213 | 2,655 | **45.69%** |
| **9** (End) | 1,192 | 2,655 | **44.90%** |

![10-Document Context Retrieval Graph](./results/10documents_results/kaggle_graph_llama3_zoomed_no_baseline.png)

* **Observations:**
  * **Highest Baseline:** Peak performance of **49.49%** is achieved when the gold document is at the very beginning (Position 0).
  * **Monotonic Decline:** Performance drops steadily as the document moves deeper into the context window, falling to **45.69%** at the middle (Position 4) and bottoming out at **44.90%** at the end (Position 9).
  * **No Recency Uptick:** Unlike the 20 and 30 document configurations, there is no recency bias effect (uptick in accuracy at the final position) for the 10-document configuration.

---

### 20-Document Context Configuration
* **Total Run Time:** 13.18 Hours
* **Total Queries:** 13,275 (2,655 questions × 5 positions)

| Gold Position (Index) | Correct Predictions | Total Evaluated | Accuracy (%) |
| :---: | :---: | :---: | :---: |
| **0** (Beginning) | 1,270 | 2,655 | **47.83%** |
| **4** (Early-mid) | 1,124 | 2,655 | **42.34%** |
| **9** (Middle) | 1,122 | 2,655 | **42.26%** |
| **14** (Late-mid) | 1,119 | 2,655 | **42.15%** |
| **19** (End) | 1,136 | 2,655 | **42.79%** |

![20-Document Context Retrieval Graph](./results/20documents_results/lost_in_the_middle_20_docs_final.png)

* **Observations:**
  * **Degraded Baseline:** Peak accuracy at the beginning drops to **47.83%** (a 1.66% decline compared to 10 documents).
  * **The "U-Shape" Curve:** Performance drops to its lowest point in the late-middle context window at **42.15%** (Position 14).
  * **Recency Bias Recovery:** Accuracy increases to **42.79%** at the final position (Position 19), showing the classic recency effect where the model recalls information from the end of the context better than the middle.

---

### 30-Document Context Configuration
* **Total Run Time:** 28.4 Hours
* **Total Queries:** 18,585 (2,655 questions × 7 positions)

| Gold Position (Index) | Correct Predictions | Total Evaluated | Accuracy (%) |
| :---: | :---: | :---: | :---: |
| **0** (Beginning) | 1,229 | 2,655 | **46.29%** |
| **4** (5th) | 1,091 | 2,655 | **41.09%** |
| **9** (10th) | 1,102 | 2,655 | **41.51%** |
| **14** (15th) | 1,085 | 2,655 | **40.87%** |
| **19** (20th) | 1,082 | 2,655 | **40.75%** |
| **24** (25th) | 1,082 | 2,655 | **40.75%** |
| **29** (End) | 1,120 | 2,655 | **42.18%** |

![30-Document Context Retrieval Graph](./results/30documents_results/lost_in_the_middle_30_docs_final.png)

* **Observations:**
  * **Lowest Baseline:** Peak accuracy at the beginning is at its lowest at **46.29%** (a 3.2% decline from the 10-document run).
  * **Extended Performance Valley:** Accuracy floors and plateaus at **40.75%** in the late-middle positions (Positions 19 and 24).
  * **Strong Recency Bounce:** The final document position (Position 29) shows a prominent recovery to **42.18%** (a 1.43% improvement from the floor), indicating that recency bias remains a significant factor as context window length increases.

---

## 2. Findings & Comparison Analysis

> **💡 Tip:** Comparing all three configurations highlights that context size has a direct, negative correlation with retrieval accuracy. The performance "valley" deepens and widens as context sizes expand.

```mermaid
graph TD
    A[Increase context size from 10 to 30 docs] --> B[Overall retrieval baseline drops by 3.2%]
    A --> C[U-Shape curve widens]
    A --> D[Floor accuracy drops from 44.9% to 40.75%]
```

The data shows that for search and question-answering systems running on `llama3.1:8b`, keeping the number of returned search results as close to **10 documents** as possible is recommended to maintain peak recall accuracy. Passing 30 documents degraded performance across all context slots.
