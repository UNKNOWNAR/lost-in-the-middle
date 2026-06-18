# Lost-in-the-Middle (LitM) Non-Factoid RAG Pipeline Progress Log

This document tracks the progress of the LitM RAG pipeline execution, including any deviations from the original plan and the rationale behind those design decisions.

---

## Phase 1: Data Preparation

**Objective:** Parse the raw TREC RAGgy-dev data (`bm25.top1000.raggy-dev.jsonl`) to build the initial pool of golden and distractor passages per query.

### Work Completed:
1. **Schema Correction:** Adapted the data parsing script (`01_data_preparation.py`) to correctly handle the structure of the JSONL file, which natively contains the `qrels` dictionaries and top 1,000 candidates with passage suffixes (e.g., `#3_2707237772`).
2. **Golden Document Search Scope:** 
   - *Original Plan:* Sift only the top 100 BM25 documents per query for golden passages (relevance $\ge$ 2).
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
- **Golden Docs (Qrels $\ge$ 2):** Average ~91.6
- **Distractor Docs (Qrels $<$ 2 or not present):** Average ~908.4

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
*   **Output File:** `litm_pipeline/data/processed/nuggets.json`

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
*   **Output File:** `litm_pipeline/data/processed/nugget_doc_alignment.json`

---

## Phase 4: Context Construction

*Log will be updated as we proceed through the pipeline phases.*

---

## Phase 5: Answer Generation

**Objective:** Generate answers for the queries using Llama and Qwen models on Kaggle.

### Work Completed / Decisions Made:
1. **Model & Platform:** Generated `k20_1` outputs on Kaggle using T4 x2 instances. We used `vLLM` to accelerate generation.
2. **Context Window Configuration:** Due to Kaggle T4 limitations, we successfully ran generation for the `k20_1` condition.
3. **Bug Fixes:** Injected monkey-patches to resolve a vLLM configuration bug regarding `rope_scaling` factors which broke the models on T4.

---

## Phase 6: Evaluation

**Objective:** Evaluate the quality of the generated answers by checking if they contain the required nuggets and measuring hallucination.

### Work Completed / Decisions Made:
1. **Custom LLM-as-a-Judge script (Nuggets):** We built `06_evaluate_answers.py` because the new "Nugget-based" data structure differs from the exact-match approach of the original pipeline.
2. **Custom LLM-as-a-Judge script (Support):** We built `07_evaluate_support.py` to evaluate the **Hallucination Index**. It passes the full 20-document context and the model answer to Gemini 1.5 Flash, decomposing answers into statements and classifying each as Fully Supported, Partially Supported, or Unsupported.
3. **Implementation Details:**
   - Evaluator queries the Gemini API (`gemini-3.1-flash-lite`) using strict prompts to detect factual overlap and evaluate support.
   - Built-in rate limiting handles `429` (Quota exceeded) and `503` (Service Unavailable) limits automatically by pausing and retrying to abide by the 15 RPM limit.
4. **Output format:** Results are recorded as JSON containing detailed query-level mappings of exactly which nuggets/statements were covered/supported.

---

## Phase 7: Analysis & Visualization

*Log will be updated as we proceed through the pipeline phases.*
