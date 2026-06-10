# LitM RAG Study: Granular Task Checklist

This checklist breaks down the execution of the **Lost-in-the-Middle (LitM) Non-Factoid RAG pipeline** into small, actionable steps.

---

## Phase 1: Data Preparation (`01_data_preparation.py`)

This phase parses the raw TREC RAGgy-dev data and maps out which queries and passages we are focusing on.

- [ ] **Task 1.1: Load Raw JSONL Data**
  - Read `litm_pipeline/data/raw/bm25.top1000.raggy-dev.jsonl` line by line.
  - Parse the JSON object from each line (contains queries and their corresponding top-1000 BM25 retrieved passages).
- [ ] **Task 1.2: Extract Queries**
  - Extract the unique query ID (`query_id`) and query text (`query`) from the raw file.
  - Store them as a key-value dictionary (e.g., `{"query_id": "query_text"}`).
  - Save to `litm_pipeline/data/processed/queries.json`.
- [ ] **Task 1.3: Extract Top-100 Passages per Query**
  - For each query, slice the top 100 passages from the ranked list.
  - Extract fields: `docid`, `title`, `text`, and BM25 `score` / `rank`.
  - Save to `litm_pipeline/data/processed/per_query_passages.json`.
- [ ] **Task 1.4: Parse Qrels (Relevance Judgments)**
  - Read `litm_pipeline/data/raw/qrels.umbrela.rag24.test.txt` (which has columns: `query_id`, `iteration`, `doc_id`, `relevance_score`).
  - Map which documents are judged highly relevant for each query (relevance score $\ge 1$ or $\ge 2$).
  - Save this mapping to `litm_pipeline/data/processed/qrels_map.json` to help identify gold/relevant documents.

---

## Phase 2: Nugget Extraction (`02_nugget_creation.py` & `prompts/nugget_creation_prompt.txt`)

Since we are evaluating non-factoid answers, we need "nuggets" (atomic units of information) to score them.

- [ ] **Task 2.1: Define Nugget Extraction Prompt**
  - Create the LLM prompt template in `litm_pipeline/prompts/nugget_creation_prompt.txt`.
  - The prompt asks the LLM to read a query and reference answers/relevant passages, then extract a list of 5–10 distinct, atomic, non-overlapping information nuggets.
- [ ] **Task 2.2: Implement Nugget Extraction Script**
  - Write a Python script using the Gemini API.
  - Load queries from `queries.json` and relevance judgments from `qrels_map.json`.
  - For each of the 120 queries, select the top 3 gold (relevance $\ge 2$) passages to act as references.
  - Call Gemini Flash to generate the nuggets.
- [ ] **Task 2.3: Handle Free-Tier Rate Limits (Batching)**
  - Implement a 15 RPM (requests per minute) rate-limiter wrapper (e.g., `time.sleep(4)` between calls) so the script doesn't crash on free tier.
- [ ] **Task 2.4: Save Extracted Nuggets**
  - Parse the JSON/list response from Gemini.
  - Save the structured nuggets to `litm_pipeline/data/processed/nuggets.json` in the format:
    ```json
    {
      "query_id": [
        {"nugget_id": "nugget_0", "text": "Nugget description"},
        {"nugget_id": "nugget_1", "text": "Another nugget description"}
      ]
    }
    ```

---

## Phase 3: Nugget-to-Document Alignment (`03_nugget_doc_alignment.py` & `prompts/nugget_assignment_prompt.txt`)

We must know which retrieved documents contain which nuggets to place them strategically in the context window.

- [ ] **Task 3.1: Define Alignment Prompt**
  - Create the alignment prompt in `litm_pipeline/prompts/nugget_assignment_prompt.txt`.
  - The prompt asks the LLM: "Does document X contain Nugget Y? Answer YES or NO."
- [ ] **Task 3.2: Map Nuggets against Top 100 Passages**
  - Load `nuggets.json` and `per_query_passages.json`.
  - For each query, take its nuggets and check them against the top 100 passages.
  - To save API costs, first run a fast heuristic/keyword match or use Gemini in batch to test only relevant-labeled documents.
  - Create a mapping table: `(query_id, docid) -> [list of nugget_ids contained]`.
  - Save this mapping to `litm_pipeline/data/processed/nugget_doc_alignment.json`.

---

## Phase 4: Context Construction (`04_context_builder.py`)

Here we construct the context windows ($k=20$ documents) for the three experimental conditions: Primacy, Middle, and Recency.

- [ ] **Task 4.1: Identify Nugget-Rich vs. Noise Documents**
  - For each query, look at the alignment data.
  - Select the top $M$ "gold" documents (documents containing the most nuggets). Let's use $M = 3$ nugget-dense documents.
  - Select the remaining $k - M$ (e.g., $20 - 3 = 17$) documents from the lower-ranked, non-relevant passages to act as "distractors" (noise).
- [ ] **Task 4.2: Construct Condition A (Primacy)**
  - Place the $M$ gold documents at the **beginning** of the context (ranks 1–3).
  - Place the 17 distractor documents in ranks 4–20.
  - Save to `litm_pipeline/data/processed/contexts/k20/condition_A.jsonl`.
- [ ] **Task 4.3: Construct Condition B (Middle)**
  - Place the $M$ gold documents in the **middle** of the context (e.g., ranks 9–11).
  - Place the distractor documents in ranks 1–8 and 12–20.
  - Save to `litm_pipeline/data/processed/contexts/k20/condition_B.jsonl`.
- [ ] **Task 4.4: Construct Condition C (Recency)**
  - Place the $M$ gold documents at the **end** of the context (ranks 18–20).
  - Place the distractor documents in ranks 1–17.
  - Save to `litm_pipeline/data/processed/contexts/k20/condition_C.jsonl`.

---

## Phase 5: Answer Generation on Kaggle (`notebooks/litm_llama3_k20.ipynb` or `src/05_generate_answers.py`)

We generate answers using Llama 3.1 8B Instruct. We'll run this on Kaggle using T4 GPUs.

- [ ] **Task 5.1: Write Kaggle Generation Notebook**
  - Build `notebooks/litm_llama3_k20.ipynb` which:
    - Loads the HuggingFace Llama 3.1 8B Instruct model using HuggingFace `transformers` or `vllm`.
    - Reads the context files for Conditions A, B, and C.
    - Uses the prompt template in `prompts/generation_prompt.txt` to instruct the model to answer the query *only using the provided passages*.
- [ ] **Task 5.2: Execute Notebook on Kaggle**
  - Upload the notebook and processed context files to Kaggle.
  - Run inference over the 120 queries $\times$ 3 conditions = 360 prompt inputs.
- [ ] **Task 5.3: Retrieve Generated Answers**
  - Download the output JSONL files from Kaggle and place them in:
    - `litm_pipeline/data/outputs/generated_answers/llama/k20_A.jsonl`
    - `litm_pipeline/data/outputs/generated_answers/llama/k20_B.jsonl`
    - `litm_pipeline/data/outputs/generated_answers/llama/k20_C.jsonl`

---

## Phase 6: Evaluation (`06_evaluate_nuggets.py` & `07_evaluate_support.py`)

Evaluate the generated answers using Gemini Flash as the judge, accounting for rate limits.

- [ ] **Task 6.1: Evaluate Nugget Recall (`06_evaluate_nuggets.py`)**
  - For each generated answer, prompt Gemini Flash: "Does the generated answer contain Nugget X? Answer YES or NO."
  - Loop over all 360 generated answers.
  - Implement batching/delay (15 RPM limit) to avoid API limit errors.
  - Calculate Recall: $\frac{\text{Number of nuggets generated}}{\text{Total nuggets for the query}}$.
  - Save results to `litm_pipeline/data/outputs/scores/nugget_scores.json`.
- [ ] **Task 6.2: Evaluate Support / Hallucination (`07_evaluate_support.py`)**
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
