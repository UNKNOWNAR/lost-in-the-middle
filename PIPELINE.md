# MASTER PIPELINE: Lost-in-the-Middle in Non-Factoid RAG
## Complete Build Specification

---

## 1. Overview

This pipeline investigates positional bias ("Lost-in-the-Middle") in long-context
non-factoid Retrieval-Augmented Generation. The hypothesis:

> When nugget-rich (evidence-dense) documents are buried in the MIDDLE of a long
> context, LLMs generate significantly worse answers than when those same documents
> sit at the START or END — even though total available information is identical.

**Dataset:** TREC 2024 RAG Track (RAGgy-dev, 120 non-factoid queries)  
**Generator:** Llama 3.1 8B Instruct (Phase 1), Qwen 2.5 7B Instruct (Phase 2)  
**Context size:** k=20 documents (Phase 1), k=30 (Phase 2)  
**Conditions:** A (Primacy), B (Middle), C (Recency)  
**Evaluation:** Gemini Flash (free tier) for nugget + support scoring  
**Supplementary:** BERTScore, ROUGE-L, optional Prometheus 2  

---

## 2. Repository Structure

```
litm_pipeline/
│
├── PIPELINE.md                        ← this file
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                           # downloaded files (gitignored)
│   │   ├── bm25.top1000.raggy-dev.jsonl
│   │   ├── qrels.rag24.test-umbrela-all.txt
│   │   └── topics.raggy-dev.tsv
│   │
│   ├── processed/
│   │   ├── queries.json
│   │   ├── per_query_passages.json
│   │   ├── nuggets.json
│   │   ├── nugget_doc_alignment.json
│   │   └── contexts/
│   │       ├── k20/
│   │       │   ├── condition_A.jsonl
│   │       │   ├── condition_B.jsonl
│   │       │   └── condition_C.jsonl
│   │       └── k30/                   # Phase 2
│   │           ├── condition_A.jsonl
│   │           ├── condition_B.jsonl
│   │           └── condition_C.jsonl
│   │
│   └── outputs/
│       ├── generated_answers/
│       │   ├── llama/
│       │   │   ├── k20_A.jsonl
│       │   │   ├── k20_B.jsonl
│       │   │   └── k20_C.jsonl
│       │   └── qwen/                  # Phase 2
│       │       └── ...
│       │
│       └── scores/
│           ├── nugget_scores.json
│           ├── support_scores.json
│           └── final_results.csv
│
├── src/
│   ├── utils.py
│   ├── 01_data_preparation.py
│   ├── 02_nugget_creation.py
│   ├── 03_nugget_doc_alignment.py
│   ├── 04_context_builder.py
│   ├── 05_generate_answers.py
│   ├── 06_evaluate_nuggets.py
│   ├── 07_evaluate_support.py
│   └── 08_analyze_results.py
│
├── notebooks/
│   └── litm_llama3_k20.ipynb
│
├── prompts/
│   ├── generation_prompt.txt
│   ├── nugget_creation_prompt.txt
│   ├── nugget_assignment_prompt.txt
│   └── support_evaluation_prompt.txt
│
└── results/
    ├── tables/
    └── figures/
```

---

## 3. Environment

### 3.1 requirements.txt

```
# Data & I/O
pandas>=2.0.0
numpy>=1.26.0
jsonlines>=4.0.0
tqdm>=4.66.0
python-dotenv>=1.2.2

# Evaluation — Gemini API (free tier)
google-genai>=2.7.0

# Embeddings — local, for nugget-doc alignment
sentence-transformers>=2.7.0

# NLP
nltk>=3.8.0

# Analysis & plotting
scipy>=1.12.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Automated metrics
bert-score>=0.3.13
rouge-score>=0.1.2

# Kaggle / generation (pre-installed on Kaggle)
# torch>=2.0.0
# transformers>=4.40.0
# accelerate>=0.27.0
# bitsandbytes>=0.43.0
```

### 3.2 .env.example

```
GEMINI_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3.3 .gitignore

```
data/raw/
data/outputs/generated_answers/
*.pyc
__pycache__/
.env
venv/
```

---

## 4. Data Sources

| File | URL | Size |
|------|-----|------|
| `bm25.top1000.raggy-dev.jsonl` | `https://huggingface.co/datasets/LDI-lab/trec-rag-2024` | ~485 MB |
| `qrels.rag24.test-umbrela-all.txt` | same HF repo | ~2.2 MB |
| `topics.raggy-dev.tsv` | same HF repo or `https://trec-rag.github.io` | small |

Download all three into `data/raw/` before running anything.

---

## 5. Shared Utilities

### File: `src/utils.py`

```python
"""
Shared utilities for the entire pipeline.
"""
import os
import re
import json
import time
import string
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ─── paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # litm_pipeline/
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS   = ROOT / "data" / "outputs"
PROMPTS_DIR    = ROOT / "prompts"
RESULTS_DIR    = ROOT / "results"

# ─── config ──────────────────────────────────────────────────────────
load_dotenv(ROOT.parent / ".env")        # reuse existing .env at repo root
load_dotenv(ROOT / ".env", override=True) # or pipeline-local .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_RPM     = 14  # safety margin on 15 RPM free-tier limit


# ─── normalize_answer (from existing codebase) ──────────────────────
def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization."""
    s = s.lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # fix whitespace
    s = " ".join(s.split())
    return s


# ─── Gemini client with rate limiting ────────────────────────────────
class GeminiClient:
    """
    Wrapper around google-genai with:
      - automatic rate limiting (free tier: 15 RPM)
      - retry with exponential backoff on 429/500
      - JSON parsing with fallback
    """

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        self._last_call = 0.0
        self._min_interval = 60.0 / GEMINI_RPM  # seconds between calls

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        expect_json: bool = False,
        max_retries: int = 3,
    ) -> str:
        """Send a prompt to Gemini, return the text response."""
        for attempt in range(max_retries):
            # rate limit
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json" if expect_json else None,
                    ),
                )
                self._last_call = time.time()
                text = response.text.strip()
                return text

            except Exception as e:
                wait = 2 ** (attempt + 1)
                logging.warning(f"Gemini API error (attempt {attempt+1}): {e}. "
                                f"Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"Gemini API failed after {max_retries} retries")

    def generate_json(self, prompt: str, **kwargs) -> dict | list:
        """Send a prompt, parse the response as JSON."""
        text = self.generate(prompt, expect_json=True, **kwargs)
        # strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


# ─── checkpointing ──────────────────────────────────────────────────
def save_checkpoint(data: dict | list, filepath: Path):
    """Save data as JSON with atomic write."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(filepath)


def load_checkpoint(filepath: Path) -> dict | list | None:
    """Load checkpoint if it exists, else return None."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── logging ─────────────────────────────────────────────────────────
def setup_logging(name: str) -> logging.Logger:
    """Consistent logging for all pipeline steps."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


# ─── prompt loading ──────────────────────────────────────────────────
def load_prompt(name: str) -> str:
    """Load a prompt template from prompts/ directory."""
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


# ─── citation extraction ────────────────────────────────────────────
def extract_citations(answer_text: str) -> list[int]:
    """Extract [Doc N] citation positions from answer text."""
    matches = re.findall(r"\[Doc\s*(\d+)\]", answer_text, re.IGNORECASE)
    return sorted(set(int(m) for m in matches))
```

---

## 6. Prompt Templates

### 6.1 `prompts/generation_prompt.txt`

```
You are a research assistant. Read all the following documents carefully and write a comprehensive answer to the question below.

Important instructions:
- Your answer must be approximately 400 words
- Synthesize information across multiple documents
- Every claim you make must be grounded in the provided documents
- Do not use any knowledge outside the provided documents
- Cite which document number(s) support each key claim, e.g. [Doc 3]

QUESTION: {query_text}

DOCUMENTS:
{rendered_context}

Write your comprehensive answer now:
```

### 6.2 `prompts/nugget_creation_prompt.txt`

```
You are a fact extraction specialist. Given a query and a set of relevant documents, extract all key information nuggets that a comprehensive answer to the query MUST cover.

A nugget is:
- One discrete, verifiable fact or argument (one sentence max)
- Grounded in the provided documents only
- Directly relevant to answering the query

For each nugget assign a vitality label:
- "vital": Without this nugget, the answer is seriously incomplete
- "okay": Useful supporting detail but not essential

QUERY: {query_text}

SOURCE DOCUMENTS:
{golden_passages_text}

Return ONLY a JSON array, no other text:
[
  {{"id": 1, "text": "nugget text here", "vitality": "vital"}},
  {{"id": 2, "text": "nugget text here", "vitality": "okay"}}
]

Rules:
- Extract between 6 and 15 nuggets depending on query complexity
- Do NOT repeat the same fact in different words
- Each nugget must be traceable to at least one source document
```

### 6.3 `prompts/nugget_assignment_prompt.txt`

```
You are an expert evaluator. For each nugget below, check whether it appears in the generated answer.

Use ONLY the information provided. Do not use any outside knowledge.

QUERY: {query_text}

GENERATED ANSWER:
"{answer_text}"

NUGGETS TO CHECK:
{numbered_nuggets}

For EACH nugget, assign a score:
- 1.0 = The nugget is clearly and accurately present in the answer
- 0.5 = The nugget is implied or partially addressed
- 0.0 = The nugget is absent or contradicted

Return ONLY a JSON array, one entry per nugget:
[
  {{"nugget_id": 1, "score": 1.0, "reason": "one sentence explanation"}},
  {{"nugget_id": 2, "score": 0.0, "reason": "one sentence explanation"}}
]
```

### 6.4 `prompts/support_evaluation_prompt.txt`

```
You are a strict fact-checker. Your ONLY source of knowledge is the documents provided below. You have NO other knowledge whatsoever.

If a claim is not explicitly stated in these documents, it is NOT supported — even if you know it to be true from elsewhere.

SENTENCES TO CHECK:
{numbered_sentences}

SOURCE DOCUMENTS:
{all_context_docs}

For EACH sentence, determine whether it is supported by the documents above.
- 1.0 = Fully supported — directly stated in at least one document
- 0.5 = Partially supported — related content exists but not directly entailed
- 0.0 = Not supported — contradicts documents or introduces external knowledge

Return ONLY a JSON array, one entry per sentence:
[
  {{"sentence_id": 1, "score": 1.0, "reason": "one sentence", "supporting_doc": "Doc 3"}},
  {{"sentence_id": 2, "score": 0.0, "reason": "one sentence", "supporting_doc": null}}
]
```

---

## 7. Pipeline Steps

---

### STEP 1 — Data Preparation
**File:** `src/01_data_preparation.py`  
**Runs:** locally, no GPU, ~2 minutes  
**Input:** 3 files in `data/raw/`  
**Output:** `data/processed/queries.json`, `data/processed/per_query_passages.json`

```python
"""
Parse TREC 2024 RAGgy-dev data into per-query passage pools.

────────────────────────────────────────────────────────────
INPUT FILES
────────────────────────────────────────────────────────────

1) topics.raggy-dev.tsv
   Tab-separated: query_id <TAB> query_text
   Example:
     2027497	how does branding benefit consumers and marketers

2) qrels.rag24.test-umbrela-all.txt
   TREC qrels format: query_id 0 docid grade
   Grades: 3=perfectly relevant, 2=highly relevant, 1=related, 0=not relevant
   Example:
     2027497 0 msmarco_v2.1_doc_15_390195898#2_1157050797 3
     2027497 0 msmarco_v2.1_doc_15_778654661#7_1490633677 0

3) bm25.top1000.raggy-dev.jsonl
   One JSON object per line. Each line = one query with up to 1000 BM25 candidates.
   Schema:
     {
       "query": {"id": "2027497", "text": "how does branding benefit consumers"},
       "candidates": [
         {
           "docid": "msmarco_v2.1_doc_...",
           "score": 18.29,
           "doc": {
             "url": "...",
             "title": "...",
             "segment": "..."    # full passage text
           }
         }
       ]
     }

────────────────────────────────────────────────────────────
PROCESSING LOGIC
────────────────────────────────────────────────────────────

1. Load topics → dict {query_id: query_text}
2. Load qrels  → dict {query_id: {docid: grade}}
3. For each line in the BM25 JSONL:
   a. Get query_id from line["query"]["id"]
   b. For each candidate in line["candidates"]:
      - Look up qrel grade (default 0 if not in qrels)
      - Record: docid, grade, bm25_score, bm25_rank, title, text
   c. Split candidates:
      - golden_passages:     grade >= 2, keep ALL
      - distractor_passages: grade <= 1, take top 50 by BM25 score
   d. If len(golden_passages) < 3:
      - Log: "SKIP query {query_id}: only {n} golden docs"
      - Skip this query
   e. Else: add to output

4. Save results

────────────────────────────────────────────────────────────
OUTPUT FILES
────────────────────────────────────────────────────────────

data/processed/queries.json:
  {
    "2027497": "how does branding benefit consumers and marketers",
    "2027498": "...",
    ...
  }

data/processed/per_query_passages.json:
  {
    "2027497": {
      "query_text": "how does branding benefit consumers and marketers",
      "golden_passages": [
        {
          "docid": "msmarco_v2.1_doc_15_390195898#2_1157050797",
          "qrel": 3,
          "bm25_score": 14.2,
          "bm25_rank": 4,
          "title": "Branding and Consumer Behavior",
          "text": "Branding helps consumers identify..."
        },
        ...
      ],
      "distractor_passages": [
        {
          "docid": "msmarco_v2.1_doc_15_778654661#7_1490633677",
          "qrel": 0,
          "bm25_score": 16.1,
          "bm25_rank": 1,
          "title": "Marketing Strategies Overview",
          "text": "Companies use various marketing..."
        },
        ...
      ],
      "n_golden": 6,
      "n_distractors": 50
    }
  }

────────────────────────────────────────────────────────────
SUMMARY STATISTICS TO LOG
────────────────────────────────────────────────────────────
- Total queries in topics file
- Total queries with qrels
- Total queries in BM25 JSONL
- Queries kept (>= 3 golden docs)
- Queries skipped (< 3 golden docs), list their IDs
- Distribution of golden doc counts: min, median, max
- Distribution of distractor counts: min, median, max
"""
```

---

### STEP 2 — Nugget Creation
**File:** `src/02_nugget_creation.py`  
**Runs:** locally, ~10 minutes (120 Gemini API calls)  
**Input:** `data/processed/per_query_passages.json`  
**Output:** `data/processed/nuggets.json`

```python
"""
Extract information nuggets from golden documents using Gemini Flash.

────────────────────────────────────────────────────────────
PROCESS
────────────────────────────────────────────────────────────

For each query:
  1. Concatenate all golden passage texts, separated by "---"
     Prefix each with: "[Golden Doc {i}] Title: {title}\n{text}"
  2. Load nugget_creation_prompt.txt
  3. Fill template: {query_text}, {golden_passages_text}
  4. Send to Gemini Flash:
       model:       gemini-2.0-flash
       temperature: 0
       max_tokens:  800
       expect_json: True
  5. Parse JSON array response
  6. Validate:
       - Is it a list? If not, retry once
       - Each item has "id", "text", "vitality"
       - vitality is "vital" or "okay"
       - Length is 6-15
       - No duplicate texts (normalize + compare)
  7. Split into vital_nuggets and okay_nuggets
  8. Save checkpoint after every 10 queries

────────────────────────────────────────────────────────────
GEMINI API SETTINGS
────────────────────────────────────────────────────────────
  model:       gemini-2.0-flash
  temperature: 0
  max_tokens:  800
  rate_limit:  14 RPM (via GeminiClient)

  Total calls: ~120 (one per query)
  Time: ~120 calls / 14 RPM ≈ 9 minutes
  Cost: $0

────────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────────

data/processed/nuggets.json:
  {
    "2027497": {
      "query_text": "how does branding benefit consumers and marketers",
      "all_nuggets": [
        {"id": 1, "text": "Branding reduces search costs...", "vitality": "vital"},
        {"id": 2, "text": "Strong brands command premium...", "vitality": "vital"},
        {"id": 3, "text": "Brand loyalty creates switching...", "vitality": "okay"},
        ...
      ],
      "vital_nuggets": [
        {"id": 1, "text": "Branding reduces search costs...", "vitality": "vital"},
        {"id": 2, "text": "Strong brands command premium...", "vitality": "vital"}
      ],
      "okay_nuggets": [
        {"id": 3, "text": "Brand loyalty creates switching...", "vitality": "okay"}
      ],
      "total_count": 11,
      "vital_count": 6
    }
  }

────────────────────────────────────────────────────────────
LOGGING
────────────────────────────────────────────────────────────
- Per query: query_id, n_golden_docs_used, n_nuggets_extracted, n_vital
- Overall: total nuggets, mean/min/max per query, any failures
"""
```

---

### STEP 3 — Nugget-to-Document Alignment
**File:** `src/03_nugget_doc_alignment.py`  
**Runs:** locally, ~5 minutes (embedding ~1500 texts)  
**Input:** `data/processed/nuggets.json`, `data/processed/per_query_passages.json`  
**Output:** `data/processed/nugget_doc_alignment.json`

```python
"""
Map each nugget to its source document via embedding similarity.
Compute nugget concentration per golden document.

────────────────────────────────────────────────────────────
PROCESS
────────────────────────────────────────────────────────────

1. Load SentenceTransformer("BAAI/bge-base-en-v1.5")
   - ~420 MB download on first run
   - Runs fine on CPU

2. For each query:
   a. Collect all vital nugget texts
   b. Collect all golden passage texts + their docids
   c. Encode nuggets → numpy array (n_nuggets, 768)
   d. Encode passages → numpy array (n_passages, 768)
   e. Compute cosine similarity matrix (n_nuggets × n_passages)
   f. For each nugget:
      - best_passage_idx = argmax(sim_matrix[nugget_idx])
      - source_docid = passage_ids[best_passage_idx]
      - source_similarity = sim_matrix[nugget_idx][best_passage_idx]
   g. Count nuggets per passage → nugget concentration
   h. Rank golden docs by concentration descending

────────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────────

data/processed/nugget_doc_alignment.json:
  {
    "2027497": {
      "doc_nugget_counts": {
        "msmarco_v2.1_doc_ABC": 3,
        "msmarco_v2.1_doc_XYZ": 2,
        "msmarco_v2.1_doc_DEF": 1,
        "msmarco_v2.1_doc_GHI": 0
      },
      "nugget_sources": {
        "1": {"docid": "msmarco_v2.1_doc_ABC", "similarity": 0.87},
        "2": {"docid": "msmarco_v2.1_doc_ABC", "similarity": 0.91},
        "3": {"docid": "msmarco_v2.1_doc_XYZ", "similarity": 0.83}
      },
      "golden_docs_ranked_by_concentration": [
        "msmarco_v2.1_doc_ABC",
        "msmarco_v2.1_doc_XYZ",
        "msmarco_v2.1_doc_DEF",
        "msmarco_v2.1_doc_GHI"
      ]
    }
  }

────────────────────────────────────────────────────────────
LOGGING
────────────────────────────────────────────────────────────
- Per query: n_nuggets, n_golden_docs, concentration distribution
- Overall: mean similarity score, any nuggets with similarity < 0.5
"""
```

---

### STEP 4 — Context Builder
**File:** `src/04_context_builder.py`  
**Runs:** locally, ~1 minute  
**Input:** `per_query_passages.json`, `nugget_doc_alignment.json`  
**Output:** `data/processed/contexts/k20/condition_{A,B,C}.jsonl`

```python
"""
Build prompt contexts for all query × condition combinations.

────────────────────────────────────────────────────────────
PARAMETERS
────────────────────────────────────────────────────────────
  K = 20                    # total documents per context (Phase 1)
  CONDITIONS = ["A", "B", "C"]

────────────────────────────────────────────────────────────
PROCESS (for each query)
────────────────────────────────────────────────────────────

1. Load golden docs sorted by nugget concentration (descending)
2. Load distractor pool sorted by BM25 score (descending)
3. n_golden = len(golden_docs)
4. n_distractors = K - n_golden
5. If n_golden > K:
     Truncate golden to K docs (keep highest concentration)
     n_distractors = 0
6. distractors = distractor_pool[:n_distractors]

7. Build three orderings:

   Condition A — PRIMACY (golden at START):
     context = golden_docs + distractors
     Golden positions: 1..n_golden

   Condition B — MIDDLE (golden BURIED in center):
     first_half  = distractors[:n_distractors // 2]
     second_half = distractors[n_distractors // 2:]
     context = first_half + golden_docs + second_half

     IMPORTANT: Within golden_docs, KEEP concentration-descending order.
     The most nugget-rich doc sits at the deepest middle position.
     Golden positions: (n_distractors//2 + 1)..(n_distractors//2 + n_golden)

   Condition C — RECENCY (golden at END):
     context = distractors + golden_docs
     Golden positions: (n_distractors + 1)..K

8. Render each context into a prompt-ready string:

   [Document 1]
   Title: {title}
   {segment_text}
   ---
   [Document 2]
   Title: {title}
   {segment_text}
   ---
   ...

9. VERIFY for each context:
   - Total docs == K
   - Same set of documents across A, B, C
   - Golden docs have the same internal order in all conditions
   - Each doc appears exactly once

────────────────────────────────────────────────────────────
OUTPUT (one JSONL file per condition)
────────────────────────────────────────────────────────────

data/processed/contexts/k20/condition_B.jsonl (one line per query):
  {
    "query_id": "2027497",
    "query_text": "how does branding benefit consumers and marketers",
    "k": 20,
    "condition": "B",
    "n_golden": 6,
    "n_distractors": 14,
    "golden_doc_positions": [8, 9, 10, 11, 12, 13],
    "golden_doc_ids": ["msmarco_v2.1_doc_ABC", ...],
    "context_docs": [
      {
        "position": 1,
        "docid": "msmarco_v2.1_doc_QRS",
        "is_golden": false,
        "nugget_count": 0,
        "title": "...",
        "text": "..."
      },
      {
        "position": 8,
        "docid": "msmarco_v2.1_doc_ABC",
        "is_golden": true,
        "nugget_count": 3,
        "title": "...",
        "text": "..."
      },
      ...
    ],
    "rendered_context": "[Document 1]\nTitle: ...\n...\n---\n[Document 2]\n..."
  }

────────────────────────────────────────────────────────────
LOGGING
────────────────────────────────────────────────────────────
- Per query: n_golden, n_distractors, golden positions per condition
- Overall: queries processed, approx token count (chars / 4)
- Sanity check: for each query assert set(A_docids) == set(B_docids) == set(C_docids)
"""
```

---

### STEP 5 — Answer Generation
**File:** `src/05_generate_answers.py` (local script) + `notebooks/litm_llama3_k20.ipynb` (Kaggle)  
**Runs:** Kaggle T4 ×2, three sessions (~4 hrs each)  
**Input:** `data/processed/contexts/k20/condition_{A,B,C}.jsonl`  
**Output:** `data/outputs/generated_answers/llama/k20_{A,B,C}.jsonl`

```python
"""
Generate answers for all contexts using Llama 3.1 8B on Kaggle.

────────────────────────────────────────────────────────────
KAGGLE NOTEBOOK STRUCTURE (3 cells)
────────────────────────────────────────────────────────────

CELL 1 — SETUP:
  !pip install -q accelerate bitsandbytes
  import torch, json
  from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
  from pathlib import Path

  MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
  )

  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForCausalLM.from_pretrained(
      MODEL_ID,
      quantization_config=bnb_config,
      device_map="auto",
  )

CELL 2 — GENERATION:
  CONDITION = "A"  # ← change per session: "A", "B", "C"
  CONTEXT_FILE = f"/kaggle/input/litm-contexts/condition_{CONDITION}.jsonl"
  OUTPUT_FILE = f"/kaggle/working/k20_{CONDITION}.jsonl"
  CHECKPOINT_FILE = f"/kaggle/working/checkpoint_{CONDITION}.json"

  GENERATION_PROMPT_TEMPLATE = '''You are a research assistant...'''

  # Load checkpoint if resuming
  completed = set()
  if Path(CHECKPOINT_FILE).exists():
      completed = set(json.loads(Path(CHECKPOINT_FILE).read_text()))

  results = []
  with open(CONTEXT_FILE, "r") as f:
      lines = f.readlines()

  for i, line in enumerate(lines):
      entry = json.loads(line)
      query_id = entry["query_id"]

      if query_id in completed:
          continue

      # Build prompt
      prompt = GENERATION_PROMPT_TEMPLATE.format(
          query_text=entry["query_text"],
          rendered_context=entry["rendered_context"],
      )

      # Format for Llama 3.1 chat template
      messages = [{"role": "user", "content": prompt}]
      input_ids = tokenizer.apply_chat_template(
          messages, return_tensors="pt", add_generation_prompt=True
      ).to(model.device)

      # Generate
      t0 = time.time()
      with torch.no_grad():
          output_ids = model.generate(
              input_ids,
              max_new_tokens=600,
              temperature=0.3,
              do_sample=True,
              top_p=0.9,
              pad_token_id=tokenizer.eos_token_id,
          )
      gen_time = time.time() - t0

      # Decode only the new tokens
      answer = tokenizer.decode(
          output_ids[0][input_ids.shape[1]:],
          skip_special_tokens=True
      ).strip()

      # Extract citations
      import re
      cited = re.findall(r"\[Doc\s*(\d+)\]", answer, re.IGNORECASE)
      cited_positions = sorted(set(int(c) for c in cited))

      result = {
          "query_id": query_id,
          "query_text": entry["query_text"],
          "model": "llama-3.1-8b",
          "k": entry["k"],
          "condition": CONDITION,
          "n_golden": entry["n_golden"],
          "golden_doc_positions": entry["golden_doc_positions"],
          "generated_answer": answer,
          "word_count": len(answer.split()),
          "citations_found": [f"Doc {p}" for p in cited_positions],
          "cited_positions": cited_positions,
          "generation_time_seconds": round(gen_time, 2),
      }
      results.append(result)
      completed.add(query_id)

      # Checkpoint every 20 queries
      if len(completed) % 20 == 0:
          with open(OUTPUT_FILE, "w") as f:
              for r in results:
                  f.write(json.dumps(r, ensure_ascii=False) + "\n")
          Path(CHECKPOINT_FILE).write_text(json.dumps(list(completed)))
          print(f"Checkpoint: {len(completed)}/{len(lines)} done")

  # Final save
  with open(OUTPUT_FILE, "w") as f:
      for r in results:
          f.write(json.dumps(r, ensure_ascii=False) + "\n")
  print(f"Done: {len(results)} answers generated for Condition {CONDITION}")

CELL 3 — DOWNLOAD:
  from IPython.display import FileLink
  display(FileLink(OUTPUT_FILE))

────────────────────────────────────────────────────────────
EXECUTION PLAN
────────────────────────────────────────────────────────────
  Session 1: CONDITION = "A" → 120 answers → ~3-4 hours
  Session 2: CONDITION = "B" → 120 answers → ~3-4 hours
  Session 3: CONDITION = "C" → 120 answers → ~3-4 hours

  Upload context JSONL files as a Kaggle dataset before running.

  Download k20_A.jsonl, k20_B.jsonl, k20_C.jsonl after each session.
  Place them in: data/outputs/generated_answers/llama/

────────────────────────────────────────────────────────────
OUTPUT FORMAT (one line per query)
────────────────────────────────────────────────────────────
  {
    "query_id": "2027497",
    "query_text": "...",
    "model": "llama-3.1-8b",
    "k": 20,
    "condition": "A",
    "n_golden": 6,
    "golden_doc_positions": [1, 2, 3, 4, 5, 6],
    "generated_answer": "Branding plays a crucial role...",
    "word_count": 387,
    "citations_found": ["Doc 2", "Doc 5", "Doc 7"],
    "cited_positions": [2, 5, 7],
    "generation_time_seconds": 4.2
  }
"""
```

---

### STEP 6 — Nugget Evaluation (Comprehensiveness)
**File:** `src/06_evaluate_nuggets.py`  
**Runs:** locally, ~30 minutes (batched Gemini API calls)  
**Input:** generated answers + `nuggets.json` + `nugget_doc_alignment.json`  
**Output:** `data/outputs/scores/nugget_scores.json`

```python
"""
Score nugget coverage for every generated answer using Gemini Flash.

────────────────────────────────────────────────────────────
BATCHING STRATEGY
────────────────────────────────────────────────────────────
  Batch up to 7 nuggets per API call.

  For each answer:
    1. Load its query's vital nuggets
    2. Chunk into batches of 7
    3. For each batch:
       - Fill nugget_assignment_prompt.txt template
       - {numbered_nuggets} = "1. \"nugget text\"\n2. \"nugget text\"\n..."
       - Send to Gemini Flash (temperature=0, expect_json=True)
       - Parse JSON array response
    4. Merge batch results

  API CALLS:
    360 answers × ceil(7 nuggets / 7 per batch) ≈ 360 calls
    At 14 RPM: ~26 minutes
    Cost: $0

────────────────────────────────────────────────────────────
SCORE CALCULATION
────────────────────────────────────────────────────────────

  Per answer:
    C_score = sum(all nugget scores) / (n_vital_nuggets * 1.0)
    Range: [0.0, 1.0]

  Per answer, by source position:
    For each nugget, look up its source doc's position in this condition's context.
    Bin positions into thirds: start, middle, end.

    C_score_edge   = mean(scores of nuggets from start or end third)
    C_score_middle = mean(scores of nuggets from middle third)

    This is the KEY analysis:
      In Condition B: C_score_middle should be LOW
      In Condition A: C_score for start-positioned nuggets should be HIGH

────────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────────

data/outputs/scores/nugget_scores.json:
  {
    "2027497:llama-3.1-8b:k20:A": {
      "query_id": "2027497",
      "model": "llama-3.1-8b",
      "k": 20,
      "condition": "A",
      "nugget_scores": [
        {
          "nugget_id": 1,
          "nugget_text": "Branding reduces search costs...",
          "score": 1.0,
          "reason": "Answer explicitly states...",
          "source_docid": "msmarco_v2.1_doc_ABC",
          "source_position_in_context": 2,
          "position_bin": "start"
        },
        ...
      ],
      "C_score": 0.714,
      "C_score_vital_only": 0.714,
      "n_vital_nuggets": 7,
      "C_score_by_position_bin": {
        "start": 0.92,
        "middle": 0.58,
        "end": 0.83
      }
    }
  }
"""
```

---

### STEP 7 — Support Evaluation (Faithfulness)
**File:** `src/07_evaluate_support.py`  
**Runs:** locally, ~90 minutes (batched Gemini API calls)  
**Input:** generated answers + context JSONL files  
**Output:** `data/outputs/scores/support_scores.json`

```python
"""
Score sentence-level faithfulness for every generated answer.

────────────────────────────────────────────────────────────
SENTENCE SPLITTING
────────────────────────────────────────────────────────────
  import nltk
  nltk.download("punkt_tab", quiet=True)
  sentences = nltk.sent_tokenize(answer_text)
  sentences = [s for s in sentences if len(s.split()) >= 10]

  Expected: ~12-15 sentences per 400-word answer

────────────────────────────────────────────────────────────
BATCHING STRATEGY
────────────────────────────────────────────────────────────
  Batch up to 5 sentences per API call.
  Each call includes the full rendered context (all 20 docs).

  For each answer:
    1. Split answer into sentences
    2. Load the rendered_context for this query+condition
    3. Chunk sentences into batches of 5
    4. For each batch:
       - Fill support_evaluation_prompt.txt template
       - {numbered_sentences} = numbered list of sentences
       - {all_context_docs} = full rendered context from JSONL
       - Send to Gemini Flash (temperature=0, expect_json=True)
    5. Merge batch results

  NOTE: Each call has a large input (~3K-5K tokens for 20 docs + sentences).
  This is fine for Gemini Flash which accepts up to 1M tokens.

  API CALLS:
    360 answers × ceil(13 sentences / 5 per batch) ≈ 1,080 calls
    At 14 RPM: ~77 minutes
    At 1,500/day: fits in 1 day
    Cost: $0

────────────────────────────────────────────────────────────
SCORE CALCULATION
────────────────────────────────────────────────────────────

  Per answer:
    S_score = sum(sentence_scores) / (n_sentences * 1.0)
    hallucination_rate = count(score == 0.0) / n_sentences

────────────────────────────────────────────────────────────
OUTPUT
────────────────────────────────────────────────────────────

data/outputs/scores/support_scores.json:
  {
    "2027497:llama-3.1-8b:k20:A": {
      "query_id": "2027497",
      "model": "llama-3.1-8b",
      "k": 20,
      "condition": "A",
      "S_score": 0.81,
      "hallucination_rate": 0.08,
      "n_sentences": 13,
      "sentence_scores": [
        {
          "sentence_id": 1,
          "sentence": "Branding plays a crucial role...",
          "score": 1.0,
          "reason": "Directly stated in Doc 3",
          "supporting_doc": "Doc 3"
        },
        {
          "sentence_id": 5,
          "sentence": "Research shows that 60% of...",
          "score": 0.0,
          "reason": "No document mentions this statistic",
          "supporting_doc": null
        },
        ...
      ]
    }
  }
"""
```

---

### STEP 8 — Analysis & Results
**File:** `src/08_analyze_results.py`  
**Runs:** locally, ~2 minutes  
**Input:** `nugget_scores.json`, `support_scores.json`, generated answers  
**Output:** tables + figures in `results/`

```python
"""
Aggregate all scores. Produce tables, figures, statistical tests.

════════════════════════════════════════════════════════════
ANALYSIS 1 — MAIN RESULTS TABLE
════════════════════════════════════════════════════════════

Aggregate per condition (A / B / C):
  - mean C_score (comprehensiveness)
  - mean S_score (support / faithfulness)
  - mean hallucination_rate
  - mean word_count

Format:
  | Condition | C_score | S_score | Halluc. Rate | Avg Words |
  |-----------|---------|---------|--------------|-----------|
  | A (Primacy) | 0.71  | 0.81    | 0.08         | 392       |
  | B (Middle)  | 0.43  | 0.57    | 0.22         | 378       |
  | C (Recency) | 0.67  | 0.76    | 0.11         | 389       |

Save: results/tables/main_results.csv

════════════════════════════════════════════════════════════
ANALYSIS 2 — U-CURVE PLOT
════════════════════════════════════════════════════════════

X-axis: Condition (Primacy / Middle / Recency)
Y-axis: Score
Two lines: C_score, S_score
Error bars: 95% confidence interval

Expected shape: U-curve (∪) with Middle at the bottom.

Plot style: reuse seaborn pattern from existing plot_llama3_comparison.py
  - sns.lineplot with markers
  - viridis palette
  - Annotate mid-point drop: "Δ = -X.XX"

Save: results/figures/u_curve_llama.png

════════════════════════════════════════════════════════════
ANALYSIS 3 — STATISTICAL SIGNIFICANCE
════════════════════════════════════════════════════════════

For paired tests (same 120 queries across conditions):
  from scipy.stats import wilcoxon

  # C_score: A vs B
  stat_AB, p_AB = wilcoxon(C_scores_A, C_scores_B, alternative="greater")
  # C_score: C vs B
  stat_CB, p_CB = wilcoxon(C_scores_C, C_scores_B, alternative="greater")
  # S_score: same tests
  # Hallucination: B vs A
  stat_hall, p_hall = wilcoxon(halluc_B, halluc_A, alternative="greater")

  Report p-values. Apply Bonferroni correction if > 3 tests.
  Target: p < 0.05 for significance, p < 0.001 for strong significance.

Save: results/tables/statistical_tests.csv

════════════════════════════════════════════════════════════
ANALYSIS 4 — NUGGET RECOVERY BY SOURCE POSITION
════════════════════════════════════════════════════════════

For each condition, for each nugget:
  Record (position_bin, score)
  position_bin = which third of the context the source doc was in

Grouped bar chart:
  X-axis: position bin (start / middle / end)
  Y-axis: mean nugget score
  Hue: condition (A / B / C)

Expected:
  Condition A + nuggets from "start" bin → HIGH recovery (~0.9)
  Condition B + nuggets from "middle" bin → LOW recovery (~0.3)
  Condition C + nuggets from "end" bin → HIGH recovery (~0.85)

Save: results/figures/nugget_recovery_by_position.png

════════════════════════════════════════════════════════════
ANALYSIS 5 — CITATION BIAS ANALYSIS
════════════════════════════════════════════════════════════

From generated answers, extract cited_positions.

For Condition B specifically:
  Bin cited positions into thirds (start / middle / end of context).
  Count citations per bin.

  Expected: citations cluster at start and end bins even though
  golden docs are in the middle bin.
  This is MECHANISTIC evidence of positional attention bias.

Visualize as:
  Heatmap: condition × position_bin → citation frequency
  Or stacked bar chart.

  Chi-square test:
    from scipy.stats import chi2_contingency
    contingency = [[start_cites, middle_cites, end_cites],  # Condition A
                    [start_cites, middle_cites, end_cites],  # Condition B
                    [start_cites, middle_cites, end_cites]]  # Condition C
    chi2, p, dof, expected = chi2_contingency(contingency)

Save: results/figures/citation_bias_heatmap.png

════════════════════════════════════════════════════════════
ANALYSIS 6 — HALLUCINATION AS LitM SYMPTOM
════════════════════════════════════════════════════════════

Bar chart:
  X-axis: Condition (A / B / C)
  Y-axis: mean hallucination_rate
  Error bars: 95% CI

Expected: Condition B has the highest hallucination rate.
Interpretation: when the model "loses" middle evidence, it fabricates filler.

Scatter plot (optional):
  X-axis: C_score (per query)
  Y-axis: hallucination_rate (per query)
  Hue: condition
  Expected: negative correlation — lower comprehensiveness → more hallucination

Save: results/figures/hallucination_rates.png

════════════════════════════════════════════════════════════
ANALYSIS 7 — AUTOMATED METRICS (BERTScore + ROUGE-L)
════════════════════════════════════════════════════════════

Compute BERTScore and ROUGE-L for each answer against
a reference = concatenated golden passage texts.

  from bert_score import score as bert_score_fn
  from rouge_score import rouge_scorer

  # BERTScore
  P, R, F = bert_score_fn(predictions, references, lang="en", verbose=True)

  # ROUGE-L
  scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
  for pred, ref in zip(predictions, references):
      scores = scorer.score(ref, pred)

Report mean BERTScore-F1 and ROUGE-L per condition.
These are supplementary — not the primary metric — but add credibility.

Save: results/tables/automated_metrics.csv

════════════════════════════════════════════════════════════
FINAL OUTPUT SUMMARY
════════════════════════════════════════════════════════════

results/
├── tables/
│   ├── main_results.csv
│   ├── statistical_tests.csv
│   ├── automated_metrics.csv
│   └── per_query_scores.csv         # raw per-query scores for reproducibility
│
└── figures/
    ├── u_curve_llama.png
    ├── nugget_recovery_by_position.png
    ├── citation_bias_heatmap.png
    └── hallucination_rates.png
"""
```

---

## 8. Execution Summary

```bash
# ── SETUP ────────────────────────────────────────────────
cd litm_pipeline
pip install -r requirements.txt
cp .env.example .env
# → fill in GEMINI_API_KEY

# ── DOWNLOAD DATA ────────────────────────────────────────
# Place these in data/raw/:
#   bm25.top1000.raggy-dev.jsonl      (from HuggingFace LDI-lab/trec-rag-2024)
#   qrels.rag24.test-umbrela-all.txt  (from same repo)
#   topics.raggy-dev.tsv              (from same repo)

# ── RUN PIPELINE ─────────────────────────────────────────
python src/01_data_preparation.py      # ~2 min,  local,  free
python src/02_nugget_creation.py       # ~10 min, local,  free (Gemini)
python src/03_nugget_doc_alignment.py  # ~5 min,  local,  free (embeddings)
python src/04_context_builder.py       # ~1 min,  local,  free

# ── GENERATION (Kaggle) ─────────────────────────────────
# Upload context JSONL files as a Kaggle dataset
# Run notebook 3× with CONDITION = "A", "B", "C"
# Download k20_A.jsonl, k20_B.jsonl, k20_C.jsonl
# Place in data/outputs/generated_answers/llama/

python src/06_evaluate_nuggets.py      # ~30 min, local,  free (Gemini)
python src/07_evaluate_support.py      # ~90 min, local,  free (Gemini)
python src/08_analyze_results.py       # ~2 min,  local,  free
```

---

## 9. Phase 1 → Phase 2 Expansion

| What | Phase 1 | Phase 2 |
|------|---------|---------|
| Models | Llama 3.1 8B | + Qwen 2.5 7B |
| Context sizes | k=20 | + k=30 |
| Total answers | 360 | 1,440 |
| Kaggle sessions | 3 | 12 |
| Gemini eval calls | ~1,440 | ~5,760 |
| Timeline | ~3 days | ~2 weeks |

Phase 2 reuses 100% of the pipeline code. Only changes:
- `05_generate_answers.py`: swap model ID, re-run
- `04_context_builder.py`: add K=30, re-run
- `08_analyze_results.py`: add multi-model comparison plots

---

## 10. Expected Findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **U-curve in C_score** | A > C > B across conditions |
| 2 | **U-curve in S_score** | Same pattern, hallucination highest in B |
| 3 | **Nugget position effect** | Nuggets from middle-positioned docs recovered poorly in B |
| 4 | **Citation bias** | In Condition B, cited positions cluster at context edges |
| 5 | **Hallucination link** | Higher hallucination_rate correlates with lower C_score in B |
| 6 | **128K window doesn't help** | Effect present even at ~4K-8K tokens (3-6% of capacity) |

---

## 11. References

- Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL 12:157-173
- Zhang et al. (2025). Lost-in-the-Middle in Long-Text Generation. arXiv:2503.06868
- Pradeep et al. (2025). Ragnarök: A Reusable RAG Framework. ECIR 2025
- Lin et al. (2024). Initial Nugget Evaluation Results for TREC 2024 RAG. arXiv:2411.09607
- Thakur et al. (2025). Support Evaluation for TREC 2024 RAG Track. arXiv:2504.15205
