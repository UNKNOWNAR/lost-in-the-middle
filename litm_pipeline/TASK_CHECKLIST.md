# LitM RAG Study: Granular Task Checklist

This checklist breaks down the execution of the **Lost-in-the-Middle (LitM) Non-Factoid RAG pipeline** into small, actionable steps.

---

## Phase 1: Data Preparation (`01_data_preparation.py`)

This phase parses the raw TREC RAGgy-dev data and maps out which queries and passages we are focusing on.

- [x] **Task 1.1: Load Raw JSONL Data**
  - Read `litm_pipeline/data/raw/bm25.top1000.raggy-dev.jsonl` line by line.
  - Parse the JSON object from each line (contains queries and their corresponding top-1000 BM25 retrieved passages).
- [x] **Task 1.2: Extract Queries**
  - Extract the unique query ID (`query_id`) and query text (`query`) from the raw file.
  - Store them as a key-value dictionary (e.g., `{"query_id": "query_text"}`).
  - Save to `litm_pipeline/data/processed/queries.json`.
- [x] **Task 1.3: Extract Top-100 Passages per Query**
  - For each query, slice the top 100 passages from the ranked list.
  - Extract fields: `docid`, `title`, `text`, and BM25 `score` / `rank`.
  - Save to `litm_pipeline/data/processed/per_query_passages.json`.
- [x] **Task 1.4: Parse Qrels (Relevance Judgments)**
  - Read `litm_pipeline/data/raw/qrels.umbrela.rag24.test.txt` (which has columns: `query_id`, `iteration`, `doc_id`, `relevance_score`).
  - Map which documents are judged highly relevant for each query (relevance score $\ge 1$ or $\ge 2$).
  - Save this mapping to `litm_pipeline/data/processed/qrels_map.json` to help identify gold/relevant documents.

---

## Phase 2: Nugget Extraction (`02_nugget_creation.py` & `prompts/nugget_creation_prompt.txt`)

Since we are evaluating non-factoid answers, we need "nuggets" (atomic units of information) to score them.

- [x] **Task 2.1: Define Nugget Extraction Prompt**
  - Create the LLM prompt template in `litm_pipeline/prompts/nugget_creation_prompt.txt` (or embedded in script).
  - The prompt asks the LLM to read a query and reference answers/relevant passages, then extract a list of 5–10 distinct, atomic, non-overlapping information nuggets.
- [x] **Task 2.2: Implement Nugget Extraction Script**
  - Write a Python script using the LLM API (supporting both Gemini and Groq/Llama 3.3).
  - Load queries from `queries.json` and relevance judgments from `qrels_map.json`.
  - For each query, select the golden (relevance $\ge 2$) passages (capped at 20 to prevent token bloat) to act as references.
  - Call LLM API to generate the nuggets.
- [x] **Task 2.3: Handle Free-Tier Rate Limits (Batching)**
  - Implement a 14 RPM rate-limiter wrapper and rate limit retry backoffs.
- [x] **Task 2.4: Save Extracted Nuggets**
  - Parse the JSON/list response from LLM.
  - Save the structured nuggets query-by-query with atomic checkpointing.

---

## Phase 3: Nugget-to-Document Alignment (`03_nugget_doc_alignment.py` & `prompts/nugget_assignment_prompt.txt`)

We must know which retrieved documents contain which nuggets to place them strategically in the context window.

- [x] **Task 3.1: Define Alignment Prompt**
  - Create the alignment prompt in `litm_pipeline/prompts/nugget_assignment_prompt.txt`.
  - The prompt asks the LLM: "Does document X contain Nugget Y? Answer YES or NO."
- [x] **Task 3.2: Map Nuggets against Top 100 Passages**
  - Load `nuggets.json` and `per_query_passages.json`.
  - For each query, take its nuggets and check them against the top 100 passages.
  - To save API costs, first run a fast heuristic/keyword match or use Gemini in batch to test only relevant-labeled documents.
  - Create a mapping table: `(query_id, docid) -> [list of nugget_ids contained]`.
  - Save this mapping to `litm_pipeline/data/processed/nugget_doc_alignment.json`.

---

## Phase 4: Context Construction (`04_context_builder.py`)

We construct the context windows ($k=20$ documents) for the 10 sliding-window conditions (representing rank offsets: 1, 3, 5, 7, 9, 11, 13, 15, 17, 18).

- [x] **Task 4.1: Identify Nugget-Rich vs. Noise Documents**
  - Select top $M = 3$ nugget-dense "gold" documents and $20 - 3 = 17$ distractor documents per query.
- [x] **Task 4.2: Construct the 10 Experimental Conditions**
  - Slide the gold documents across the context window at different starting ranks: **1, 3, 5, 7, 9, 11, 13, 15, 17, 18**.
  - Assert that all 10 conditions use the exact same document set for each query.
  - Save to `litm_pipeline/data/processed/contexts/k20/condition_{1..10}.jsonl`.
- [x] **Task 4.3: Create Generation script & Kaggle notebook**
  - Created `src/05_generate_answers.py` and `notebooks/litm_llama3_k20_10point.ipynb` supporting vLLM batching.

---

## Phase 5: Answer Generation on Kaggle (`notebooks/litm_llama3_k20.ipynb` or `src/05_generate_answers.py`)

We generate answers using Llama 3.1 8B Instruct. We'll run this on Kaggle using T4 GPUs.

- [x] **Task 5.1: Write Kaggle Generation Notebook**
  - Build `notebooks/litm_llama3_k20.ipynb` which:
    - Loads the HuggingFace Llama 3.1 8B Instruct model using HuggingFace `transformers` or `vllm`.
    - Reads the context files for Conditions A, B, and C.
    - Uses the prompt template in `prompts/generation_prompt.txt` to instruct the model to answer the query *only using the provided passages*.
- [x] **Task 5.2: Execute Notebook on Kaggle**
  - Upload the notebook and processed context files to Kaggle.
  - Run inference over the 120 queries $\times$ 3 conditions = 360 prompt inputs.
- [x] **Task 5.3: Retrieve Generated Answers**
  - Download the output JSONL files from Kaggle and place them in:
    - `litm_pipeline/data/outputs/generated_answers/llama/k20_A.jsonl`
    - `litm_pipeline/data/outputs/generated_answers/llama/k20_B.jsonl`
    - `litm_pipeline/data/outputs/generated_answers/llama/k20_C.jsonl`

---

## Phase 6: Evaluation (`06_evaluate_nuggets.py` & `07_evaluate_support.py`)

Evaluate the generated answers using Gemini Flash as the judge, accounting for rate limits.

- [x] **Task 6.1: Evaluate Nugget Recall (`06_evaluate_nuggets.py`)**
  - For each generated answer, prompt Gemini Flash: "Does the generated answer contain Nugget X? Answer YES or NO."
  - Loop over all 360 generated answers.
  - Implement batching/delay (15 RPM limit) to avoid API limit errors.
  - Calculate Recall: $\frac{\text{Number of nuggets generated}}{\text{Total nuggets for the query}}$.
  - Save results to `litm_pipeline/data/outputs/scores/nugget_scores.json`.
- [x] **Task 6.2: Evaluate Support / Hallucination (`07_evaluate_support.py`)**
  - Prompt Gemini Flash: "For each statement in the generated answer, is it supported by the context? Answer fully supported, partially supported, or unsupported."
  - Save support scores to `litm_pipeline/data/outputs/scores/support_scores.json`.

---

## Phase 7: Analysis & Visualization (`08_analyze_results.py`)

Aggregate results and generate figures to test our hypothesis.

- [ ] **Task 7.1: Aggregate Performance Metrics**
  - Read nugget scores and support scores.
  - Calculate mean and standard deviation of Nugget Recall and Support Rate for Primacy, Middle, and Recency.
  - Save summary table to `litm_pipeline/results/tables/summary.csv`.
- [ ] **Task 7.2: Generate Performance Graphs**
  - Use `matplotlib` / `seaborn` to plot the scores.
  - x-axis: Position condition (Primacy, Middle, Recency).
  - y-axis: Nugget Recall / Support Rate.
  - Save figures to `litm_pipeline/results/figures/position_bias_k20.png`.
- [ ] **Task 7.3: Document Key Findings**
  - Write a final report or update `README.md` summarizing whether the Lost-in-the-Middle effect was observed and to what magnitude.
