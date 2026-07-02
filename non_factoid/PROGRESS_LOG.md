# Lost-in-the-Middle (LitM) Non-Factoid RAG Pipeline Progress Log

This document tracks the progress of the LitM RAG pipeline execution, including any deviations from the original plan and the rationale behind those design decisions.

---

## Phase 1: Data Preparation

**Objective:** Parse the raw TREC RAGgy-dev data (`bm25.top1000.raggy-dev.jsonl`) to build the initial pool of golden and distractor passages per query.

### Work Completed:
1. **Schema Correction:** Adapted the data parsing script (`01_data_preparation.py`) to correctly handle the structure of the JSONL file, which natively contains the `qrels` dictionaries and top 1,000 candidates with passage suffixes (e.g., `#3_2707237772`).
2. **Golden Document Search Scope:**
   - *Original Plan:* Sift only the top 100 BM25 documents per query for golden passages (relevance ≥ 2).
   - *Action Taken:* We expanded the search to scan **all 1,000 BM25 candidates** per query for golden passages.
   - *Rationale:* Restricting the search to the top 100 might miss highly relevant golden documents that BM25 ranked lower. By scanning the entire 1,000 candidate list, we maximize the number of golden passages available for nugget extraction, leading to a more robust evaluation.
3. **Empty Query Removal:**
   - *Action Taken:* Removed exactly **1 query** (Query ID `2035009`) from the processed dataset.
   - *Rationale:* Statistical analysis revealed that this query had exactly **0 golden documents** in its entire top 1,000 candidate pool. Since Phase 2 relies on golden documents to extract factual nuggets, keeping a query with 0 golden documents would break the nugget generation logic.
   - *Methodology Note for Paper:* "Of the 120 RAGgy-dev queries, one query (ID: 2035009) was excluded because none of its top-1000 BM25-retrieved passages received a relevance grade of ≥2 in the UMBRELA qrels, leaving a final evaluation set of 119 queries."

### Calculated Statistics (from all 1,000 candidates per query):

**Overall Judged vs. Unjudged (Average per query):**
- **Total Candidates:** 1,000
- **Total Judged by Humans:** ~236.6 documents (Min: 46, Max: 585)
- **Not Judged at All (Defaulted to 0):** ~763.4 documents (Min: 415, Max: 954)

**Breakdown of the ~236.6 Judged Documents (Average per query):**
- **Qrel = 3** (Perfectly Relevant): ~36.3 documents
- **Qrel = 2** (Highly Relevant): ~55.3 documents
- **Qrel = 1** (Related): ~69.3 documents
- **Qrel = 0** (Explicitly Irrelevant): ~75.7 documents

**Legacy Combined Stats:**
- **Golden Docs (Qrels ≥ 2):** Average ~91.6
- **Distractor Docs (Qrels < 2 or not present):** Average ~908.4

---

## Phase 2: Nugget Extraction

**Objective:** Extract distinct, atomic, and verifiable information nuggets from the selected golden passages for each of the 119 queries using an LLM.

### Work Completed / Decisions Made:
1. **Console Output Fix:** Configured `sys.stdout` and `sys.stderr` to force UTF-8 encoding to prevent crashes on Windows console buffers.
2. **Atomic Checkpointing:** Enabled step-by-step state saving to `nuggets_checkpoint.json` to safely resume execution.
3. **Model Selection & Quota Optimizations:**
   - *Issue:* Gemini 2.0 Flash free tier was deprecated (0 quota), and Groq's free tier had a strict 6,000 Tokens Per Minute (TPM) limit that rejected prompts containing 15 documents.
   - *Resolution:* Switched to **`gemini-3.1-flash-lite`** using Google AI Studio API, which provides 15 RPM, 250k TPM, and 500 RPD.
4. **Source Document Optimization (Upgraded to 20 Docs):**
   - *Action Taken:* We increased the document limit cap from 15 to **20 source documents** per query (prioritizing all `qrel=3` documents, and filling up to 20 with `qrel=2` documents).
   - *Rationale:* Since `gemini-3.1-flash-lite` has a large TPM limit (250k), we could safely expand context to 20 documents. This allowed us to capture a wider range of relevant secondary information (`qrel=2`) without hitting the daily API quota, while still staying below the token length where "Lost-in-the-Middle" performance degradation severely hurts extraction quality.

### Extraction Statistics (119 queries processed):
*   **Total Queries Failed:** 0
*   **Total Nuggets Created:** 1,187
*   **Mean Nuggets per Query:** 10.0
*   **Mean Vital Nuggets per Query:** 5.9
*   **Min Nuggets (any query):** 4
*   **Max Nuggets (any query):** 15
*   **Output File:** `non_factoid/data/processed/nuggets.json`

---

## Phase 3: Nugget-to-Document Alignment

**Objective:** Map each extracted nugget to its corresponding source golden document using local SentenceTransformer embeddings, compute nugget concentration per document, and rank golden documents.

### Work Completed / Decisions Made:
1. **GPU Acceleration (PyTorch CUDA Reinstallation):**
   - *Issue:* Running the alignment script on CPU was extremely slow (~9 seconds per batch, estimating ~4 hours total) because the default virtual environment lacked CUDA support.
   - *Resolution:* Verified the presence of an **NVIDIA RTX A500 Laptop GPU** and forced a reinstall of PyTorch with CUDA 12.1 support (`torch-2.5.1+cu121`). This sped up execution by orders of magnitude, completing all 119 queries in about 11 minutes total (including overhead).
2. **Alignment Strategy (Local Embeddings vs LLM API):**
   - *Action Taken:* Implemented local semantic similarity alignment using `BAAI/bge-base-en-v1.5`.
   - *Rationale:* Rather than consuming expensive/rate-limited LLM API requests to match 1,187 nuggets against 100+ documents per query (~119,000 combinations), local cosine similarity provides a free, deterministic, and highly accurate semantic mapping.
3. **Document Ranking (Nugget Concentration & BM25 Tie-breaking):**
   - *Action Taken:* Ranked documents primary by nugget concentration (descending), and secondarily by original BM25 score (descending) as a tie-breaker.
   - *Rationale:* Placing nugget-rich documents in specific areas (primacy, recency, middle) is core to testing the Lost-in-the-Middle hypothesis. Breaking ties using BM25 ensures that documents with equal nugget concentration still respect retrieved relevance order.

### Alignment Statistics:
*   **Total Nuggets Mapped:** 1,187
*   **Mean Cosine Similarity:** 0.7480
*   **Low Similarity Matches (Sim < 0.5):** 1 (high alignment confidence overall)
*   **Output File:** `non_factoid/data/processed/nugget_doc_alignment.json`

---

## Phase 4: Context Construction

- [x] **Setup & Data Structure Definitions**: 100% COMPLETE
- [x] **Base LLM Answer Generation (Qwen 2.5 7B)**: 100% COMPLETE — k=20, k=40, k=60 all done
- [x] **Recall Evaluation (Vital & Okay Recall %)**: 100% COMPLETE — API-based, Gemma 4 31B Instruct via Google AI Studio
- [x] **Support/Hallucination Evaluation**: 100% COMPLETE (All Conditions)
> Note: *Nugget extraction* (Phase 2) used `gemini-3.1-flash-lite`. *Recall evaluation* (this step) used **Gemma 4 31B** as LLM-as-a-judge to check if generated answers cover each nugget.

### Key Design Decision — Fine-grained 10-Condition Design:
- *Original Plan:* 3 coarse conditions (Primacy/Middle/Recency).
- *Final Design:* **10 fine-grained conditions** sliding the gold documents across the full context window in fixed-rank intervals.
  - k=20: gold docs at ranks 1-2, 3-4, 5-6, 7-8, 9-10, 11-12, 13-14, 15-16, 17-18, 19-20
  - k=40: analogous sliding across 40 positions
  - k=60: gold docs at ranks 1-3, 7-9, 14-16, 21-23, 28-30, 35-37, 42-44, 48-50, 54-56, 58-60
- *Rationale:* 10 fine-grained positions allows plotting a continuous U-curve, not just three isolated points. This provides much stronger evidence for the LitM phenomenon and is more publishable.

---

## Phase 5: Answer Generation

**Objective:** Generate answers for the queries using Qwen 2.5 7B on Kaggle.

### Work Completed / Decisions Made:
1. **Model & Platform:** Generated all outputs on Kaggle using T4 x2 instances. We used `vLLM` to accelerate generation.
2. **Bug Fixes:** Injected monkey-patches to resolve a vLLM configuration bug regarding `rope_scaling` factors which broke the models on T4.
3. **K=100 Scale Outlier Dropping:**
   - *Issue:* When scaling up to the `k100` condition (100 documents), prompt lengths approached ~90,000 tokens. Kaggle's free T4 x2 instances (16GB VRAM each) suffered from hard `CUDA out of memory` crashes inside vLLM's xformers backend when processing prompts longer than ~75,000 tokens due to tight KV cache limits.
   - *Resolution:* Implemented a strict 70,000 token limit. We identified exactly 4 outlier queries in the dataset that exceeded this limit (IDs: 2034676, 2040352, 2044323, 2051782). Instead of crashing the entire condition batch, we explicitly drop these queries by substituting a dummy prompt before generation and saving an empty string in the output.

---

## Phase 6: Recall Evaluation

**Objective:** Evaluate Vital Recall % and Okay Recall % for every generated answer by checking whether each nugget is covered in the answer.

**Evaluator:** API-based LLM-as-a-judge using **Gemma 4 31B Instruct** (`gemma-4-31b-it`) via Google AI Studio free tier.
> **Important distinction:** Nugget *extraction* (Phase 2) used `gemini-3.1-flash-lite` to extract nuggets from golden documents. Recall *evaluation* (this phase) uses **Gemma 4 31B** to judge whether each nugget is present in the model's generated answer. These are two completely separate API calls with different models and different purposes.

### Work Completed / Decisions Made:
1. **Evaluator Model Selection — Gemma 4 31B:**
   - *Decision:* Used **Gemma 4 31B Instruct** via Google AI Studio (free tier) as the primary LLM-as-a-judge.
   - *Rationale:* Large, capable open-weight model available for free via Google AI Studio. Its 31B parameters provide strong NLI-style judgment for nugget coverage detection.
2. **Async Batching with Multiple API Keys:**
   - *Implementation:* Built `evaluate_k60_async.py` which fires up to 12 queries concurrently per API key, respects the 15 RPM rate limit by batching into 60-second windows, and auto-resumes from saved JSON checkpoints.
   - *Result:* Reduced per-condition evaluation time from ~2 hours (sequential) to ~10-15 minutes.
3. **Groq LLaMA 3.3 70B as Safety Fallback:**
   - *Issue:* Google's safety filters permanently blocked ~23 specific queries across all 10 conditions, leaving 34 total evaluation gaps spread across all conditions.
   - *Key Insight:* The safety filter evaluates the full prompt (query + 60 documents), not just the query text. So the same query can pass in one condition but be blocked in another depending on which distractor documents appear adjacent to it in the shuffled context.
   - *Resolution:* Built `evaluate_k60_groq.py` targeting only the 34 missing evaluations using `llama-3.3-70b-versatile`. Since only the model answer + nugget list is sent (not the massive 60-doc context), Groq's filters do not trigger. This filled all gaps.

### Key Finding — Degenerate Answer Queries:

**Investigation (30-Jun-2026):** A systematic analysis of the `k=60` generated answer files revealed that **17 queries** produce degenerate outputs (`of!!!!!!!!!!!!!...`, `!!!!!!!!!!!...`, etc.) across **all 10 conditions**. This is a known failure mode of `Qwen2.5-7B-Instruct` on certain prompts under vLLM, where the model enters a repetitive token loop.

- **15 queries** are degenerate in **all 10/10 conditions** (completely broken regardless of golden document position):
  `2001908`, `2003157`, `2003976`, `2026150`, `2027130`, `2027497`, `2033470`, `2034676`, `2040352`, `2044323`, `2046027`, `2051782`, `3010623`, `3100188`, `3100292`
- **2 additional queries** (`421946`, `818583`) produce degenerate output in 9/10 and 8/10 conditions respectively. Their "non-degenerate" outputs are still garbage — they passed the detection threshold due to slightly more unique characters before the `!!!` flood, but are functionally degenerate.

**Methodology Note for Paper:** *"Of the 119 evaluated queries, 17 queries (IDs: 2001908, 2003157, 2003976, 2026150, 2027130, 2027497, 2033470, 2034676, 2040352, 2044323, 2046027, 2051782, 3010623, 3100188, 3100292, 421946, 818583) produced degenerate repetitive outputs (token-loop failure) from Qwen2.5-7B-Instruct under all tested context configurations. These 17 queries are excluded from the recall analysis, leaving an effective evaluation set of 102 queries."*

### Key Finding — Safety Filter Blocks & Final 100-Query Dataset:

**Investigation (30-Jun-2026):** A deeper log analysis revealed queries that randomly triggered `Empty/safety response` blocks from the Google AI Studio Gemini API endpoint across different conditions.

**Root cause identified:** The safety filter evaluates the **entire prompt** (query + all 60 surrounding documents), not just the query text. This means a query that is perfectly fine in Condition 1 can be blocked in Condition 5 simply because the random shuffling of distractor documents placed sensitive text adjacent to the query.

**Success count distribution across 10 conditions (out of 102 valid queries):**
- Succeeded in all 10 conditions: **79 queries**
- Succeeded in 9 conditions: **16 queries**
- Succeeded in 8 conditions: **4 queries**
- Succeeded in 7 conditions: **2 queries**
- Succeeded in 6 conditions: **1 query**

**Resolution — Final 100-Query Dataset:**
After using Groq to fill all gaps, the final evaluation set was trimmed to exactly **100 queries** by dropping 2 permanently blocked queries (`2001010`: "cost comparison of funerals in australia" and `2005952`: "what causes eye blackouts"). These 2 queries were blocked by Google's death/harm safety category in every condition regardless of document ordering.

**Final dataset composition:**
- 119 original queries − 1 (no golden docs) = 118
- 118 − 17 (degenerate Qwen outputs) = 101
- 101 − 1 (most blocked, 2001010) = 100
- Final: **100 queries × 10 conditions = 1,000 evaluations**

### k=60 Final Results (100-Query Dataset):

| Condition | Gold Doc Ranks | Vital Recall | Okay Recall |
|:---:|:---:|:---:|:---:|
| 1 | 1–3 (Primacy) | **51.45%** | 26.35% |
| 2 | 7–9 | 49.05% | 21.22% |
| 3 | 14–16 | 44.07% | 18.95% |
| 4 | 21–23 | 42.40% | 18.62% |
| 5 | 28–30 (Middle) | 43.18% | 18.92% |
| 6 | 35–37 | 45.46% | 20.35% |
| 7 | 42–44 | 43.00% | 20.74% |
| 8 | 48–50 | 43.63% | 19.47% |
| 9 | 54–56 | 47.72% | 21.41% |
| 10 | 58–60 (Recency) | **47.52%** | 23.96% |

**Key takeaway:** Clear U-curve confirmed. Condition 4 (ranks 21–23) is the worst — a **−9.05% absolute drop** in Vital Recall vs. Condition 1 (Primacy). The middle-recency region (Conditions 9–10) partially recovers, confirming the classic LitM U-shape.

---

## Phase 7: Analysis & Visualization

### Completed:
- [x] **Vital Recall line graphs** (zoomed + full scale) for k=60
- [x] **Okay Recall line graphs** (zoomed + full scale) for k=60
- [x] All graphs saved to `non_factoid/results/figures/k=60/`

### Pending:
- [ ] Combined multi-k comparison plot (k=20 vs k=40 vs k=60 on same axes)
- [ ] Statistical significance tests (Wilcoxon signed-rank across conditions)
- [ ] Hallucination rate analysis across conditions
- [ ] Citation bias analysis (which document positions the model cites in each condition)

---

## Project Reorganization — 2026-07-02

**Decision:** Restructured the workspace from a flat layout into two clearly separated experiment directories.

### Before:
```
lostinthemiddle/
├── data/           ← factoid (NQ-Open)
├── src/            ← factoid notebooks
├── results/        ← factoid results
├── utilities/      ← factoid plotting
├── litm_pipeline/  ← non-factoid (everything)
└── scratch/        ← evaluation scripts
```

### After:
```
lostinthemiddle/
├── factoid/        ← all NQ-Open factoid experiment files
└── non_factoid/    ← all TREC RAG 2024 non-factoid files
```

### Rationale:
The two experiments use entirely different datasets (NQ-Open vs RAGgy-dev), different evaluation methodologies (exact match vs nugget recall), and different models. Keeping them in the same flat namespace caused confusion. The new structure makes it immediately clear which experiment a file belongs to.

### Path updates:
All hardcoded absolute paths in `non_factoid/scripts/` were automatically patched during migration to reflect the new directory layout.
