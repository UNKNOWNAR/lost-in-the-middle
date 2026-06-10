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

### Calculated Statistics (from all 1,000 candidates per query):
**Golden Docs (Qrels $\ge$ 2):**
- Min: 0
- Max: 370
- Average: ~91.56
- Median: 63.0
- Mode: 33

**Distractor Docs (Qrels $<$ 2 or not present):**
- Min: 630
- Max: 1000
- Average: ~908.44
- Median: 937.0
- Mode: 967

---

*Log will be updated as we proceed through the pipeline phases.*
