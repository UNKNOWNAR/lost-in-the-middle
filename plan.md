# Research Plan: Lost-in-the-Middle Effect in Non-Factoid Long-Output RAG
### Bridging LONGINOUTBENCH (2025) and TREC 2024 RAG Track

---

## 1. Motivation and Gap

The "lost-in-the-middle" (LitM) phenomenon — where LLMs fail to attend to context placed in the middle of a long input — has been studied primarily in **factoid** settings: single-sentence answers, needle-in-a-haystack probes, and short QA benchmarks. The LONGINOUTBENCH paper (Zhang et al., 2025) extended this to **long-input + long-output** scenarios but used *synthetic*, domain-controlled academic paper summaries.

What remains completely unstudied is how LitM degrades generation quality in **messy, real-world, non-factoid RAG** — the dominant architecture in enterprise AI today. The TREC 2024 RAG Track provides the ideal vehicle: it uses real Bing Search query logs, subjective multi-document queries, and a community-validated evaluation infrastructure. This research directly bridges that gap.

---

## 2. Research Questions

**RQ1 (Positional Bias):** Does placing the most relevant documents in the middle of a 20–30 document context window produce a statistically significant drop in Comprehensiveness and Support scores compared to primacy and recency placement?

**RQ2 (Scale vs. Bias):** Does a 128K context window prevent LitM degradation even at modest prompt sizes (4K–8K tokens), confirming the effect is driven by attention mechanism bias, not context window limits?

**RQ3 (Model Sensitivity):** Do different open-source models (Llama 3.1 8B vs. Qwen 2.5 7B) show different susceptibility to LitM in non-factoid generation?

**RQ4 (Failure Modes):** When critical evidence is lost, do models compensate with hallucination (fabricating plausible-sounding content), omission (shorter/incomplete answers), or copying from non-relevant documents?

---

## 3. Dataset: TREC 2024 RAG Track

### 3.1 Topic Collections

The TREC 2024 RAG Track provides two non-factoid topic collections, both ideal for this research:

| Collection | Size | Description | Example Query |
|---|---|---|---|
| **TREC-RAGgy** | ~120 queries | Complex questions from TREC DL 2021–2023; require multi-source synthesis | "cost comparison of funerals in australia" |
| **TREC-Researchy** | 600 dev queries (from 96,450 total) | Multi-perspective, decompositional questions | "how does branding benefit consumers and marketers?" |

Both collections are explicitly **non-factoid**: they require long-form answers synthesized from multiple documents. The RAGgy set includes NIST qrels (relevance judgments), making it the more valuable subset for our controlled experiment.

### 3.2 Corpus

The underlying corpus is **MS MARCO V2.1 deduped segment collection**:
- 113,520,750 text passages
- Derived from 10,960,555 deduplicated documents
- Segments created via sliding window (10-sentence windows, 5-sentence stride)
- Each passage: 500–1,000 characters typically

### 3.3 Relevance Judgments (Qrels)

TREC 2024 RAGgy provides **NIST-judged qrels** with four relevance grades:
- **qrel=3**: Definitely relevant (the "golden documents" for our experiments)
- **qrel=2**: Highly relevant
- **qrel=1**: Related
- **qrel=0**: Not relevant

For our controlled experiment, documents with qrel=3 serve as the "golden" documents whose *position* we manipulate.

### 3.4 Access

Data is available from the TREC RAG website (https://trec-rag.github.io) and the Ragnarök toolkit on GitHub (https://github.com/castorini/ragnarok). The MS MARCO V2.1 corpus is downloadable courtesy of Microsoft.

---

## 4. Experimental Design

### 4.1 Context Window Construction

For each query, construct a long context of **25 documents total** (to fill approximately 4K–8K tokens, well within the 128K window but sufficient to trigger LitM):

- **5 golden documents**: qrel=3 passages for that query
- **20 distractor documents**: mixture of qrel=1 and qrel=0 passages retrieved for the same query (providing plausible but non-critical context)

Run **three positional conditions** per query:

| Condition | Document Arrangement |
|---|---|
| **A — Primacy** | Golden docs at positions 1–5 (beginning) |
| **B — Middle** | Golden docs at positions 11–15 (exact center) |
| **C — Recency** | Golden docs at positions 21–25 (end) |

The distractor documents remain in a fixed random order across conditions; only the golden documents are repositioned. This is a direct **counterfactual probe**: the only variable is position.

### 4.2 Generator Models

| Model | Parameters | Context Window | Rationale |
|---|---|---|---|
| **Llama 3.1 8B Instruct** | 8B | 128K | Meta's leading open-source model; widely deployed |
| **Qwen 2.5 7B Instruct** | 7B | 128K | Strong multilingual model; comparable scale to Llama |

Both models fit on a single A100 40GB GPU (same hardware used in LONGINOUTBENCH). Using models of similar scale allows a clean model comparison while controlling for compute.

### 4.3 Generation Prompt

Each model receives the following prompt structure:

```
You are a research assistant. Read the following documents carefully and write
a comprehensive, multi-paragraph answer (approximately 400 words) to the question below.
Your answer must be grounded in the provided documents. Cite which documents you draw upon.

QUESTION: {query}

DOCUMENTS:
[Doc 1]: {passage_text}
[Doc 2]: {passage_text}
...
[Doc 25]: {passage_text}

Write your answer now. Be thorough and cover all key aspects from the documents.
```

The 400-word target aligns with TREC 2024 RAG Track's official submission limit, making results directly comparable to track participants.

### 4.4 Sample Size

- Use all **120 RAGgy queries** (with qrel=3 judgments available)
- 3 conditions × 2 models = **720 total generated answers**
- Each answer evaluated on 2 metrics = **1,440 evaluation calls** to GPT-4o

---

## 5. Evaluation Framework

Since BLEU, ROUGE, and Exact Match all fail for long-form non-factoid generation, we deploy a **two-metric LLM-as-a-Judge** framework, adapted from both LONGINOUTBENCH (2025) and the TREC 2024 official evaluation methodology.

### 5.1 Metric 1: Comprehensiveness (Nugget Recall)

This metric asks: *Did the generated answer successfully integrate the core arguments and facts from the golden documents?*

**Method — Nugget-based Evaluation** (following AutoNuggetizer, Lin et al., 2024):

**Step 1 — Nugget Generation** (offline, once per query):
Using GPT-4o, extract 6–10 key information nuggets from the 5 golden documents. A nugget is a discrete, verifiable factual claim or argument unit.

Example for query "how does branding benefit consumers and marketers?":
- Nugget 1: "Branding reduces consumers' search costs by enabling quick identification of trusted products."
- Nugget 2: "Strong brands allow companies to charge premium prices."
- Nugget 3: "Brand loyalty creates switching costs that protect market share."

**Step 2 — Nugget Assignment** (per generated answer):
GPT-4o evaluates each nugget against the generated answer and assigns:
- **Full Support (1.0)**: Nugget is clearly present and accurately stated
- **Partial Support (0.5)**: Nugget is implied or partially addressed
- **No Support (0.0)**: Nugget is absent or contradicted

**Comprehensiveness Score:**

```
C_score = (Σ nugget_support_scores) / (total_nuggets × 1.0)
```

This produces a score in [0, 1]. Higher = more complete coverage of golden document content.

### 5.2 Metric 2: Support / Faithfulness

This metric asks: *Are the claims made in the generated answer strictly grounded in the provided documents, or did the model hallucinate?*

This directly replicates the TREC 2024 RAG Track's official "Support Evaluation" methodology (Thakur et al., 2025), which found GPT-4o achieves 56–72% agreement with human judges on this task.

**Method:**

For each **sentence** in the generated answer:

GPT-4o receives the sentence and the full set of 25 documents and assigns:
- **Fully Supported (1.0)**: Sentence is entailed by at least one provided document
- **Partially Supported (0.5)**: Sentence is plausible and related but not directly entailed
- **Not Supported (0.0)**: Sentence contradicts documents or introduces external information

**Support Score:**

```
S_score = (Σ sentence_support_labels) / (total_sentences × 1.0)
```

**GPT-4o Judge Prompt (Comprehensiveness):**

```
You are an expert evaluator. Given the question and a list of key information nuggets
extracted from gold-standard documents, determine whether each nugget is present in
the generated answer.

QUESTION: {query}

NUGGET: {nugget_text}

GENERATED ANSWER: {answer}

Score:
- 1.0 = The nugget is clearly and accurately present in the answer
- 0.5 = The nugget is partially present or implied
- 0.0 = The nugget is absent or contradicted

Return JSON: {"score": <0.0|0.5|1.0>, "reason": "<one sentence>"}
```

**GPT-4o Judge Prompt (Support):**

```
You are a fact-checking evaluator. Given a sentence from a generated answer and a set
of source documents, determine whether the sentence is supported by the documents.

ANSWER SENTENCE: {sentence}

SOURCE DOCUMENTS:
{doc_1_text}
{doc_2_text}
...

Score:
- 1.0 = Fully supported by at least one document
- 0.5 = Plausible and related, but not directly entailed
- 0.0 = Not supported, contradicts documents, or introduces external facts

Return JSON: {"score": <0.0|0.5|1.0>, "reason": "<one sentence>", "supporting_doc_id": "<doc_id or null>"}
```

### 5.3 Bonus Metric: Positional Citation Analysis

For models that produce citations (e.g., "[Doc 12]"), track which document positions are actually cited in each condition. This provides a direct mechanistic measure of attention bias — not just its downstream effect.

```
Citation_bias = frequency of citations to docs in positions 1–5
              vs positions 11–15
              vs positions 21–25
```

---

## 6. Comparison to LONGINOUTBENCH Methodology

This study extends LONGINOUTBENCH in three key ways:

| Dimension | LONGINOUTBENCH (Zhang et al., 2025) | This Study |
|---|---|---|
| **Input type** | Academic papers (clean, structured) | Real web documents (noisy, varied) |
| **Query type** | Summarization task (well-defined) | Non-factoid open questions (subjective) |
| **Output length** | 4K–16K words | ~400 words (TREC standard) |
| **Evaluation** | Length + Consistency + Quality | Nugget Recall + Support/Faithfulness |
| **Positional manipulation** | Not controlled (natural position) | Explicit counterfactual conditions A/B/C |
| **Dataset** | Synthetic (100 custom samples) | Real-world benchmark (120 NIST-judged queries) |

The LitM mitigation score from LONGINOUTBENCH (Figure 7b: consistency by paper position) is the direct analogue of our Comprehensiveness score by document position.

---

## 7. Hypothesis and Expected Results

**Primary Hypothesis:** Both Comprehensiveness and Support scores will follow a **U-shaped curve** across conditions A → B → C, with Condition B (Middle) performing worst.

**Secondary Hypotheses:**

1. The performance gap between Condition A and Condition B will be larger than the gap between Condition C and Condition B (recency bias may partially offset LitM).
2. When golden documents are in the middle (Condition B), Support scores will *decrease* (more hallucination) as models fill the answer with content from distractor documents at the edges instead.
3. Qwen 2.5 7B, trained with a stronger long-context curriculum, may show a smaller U-shape than Llama 3.1 8B.

---

## 8. Implementation Plan

### Phase 1: Data Preparation (Week 1–2)

```python
# Step 1: Download TREC RAGgy queries + qrels
# Source: https://trec-rag.github.io

# Step 2: For each query, retrieve top-50 passages from MS MARCO V2.1
# using BM25 (via Anserini/Pyserini) or a provided run file

# Step 3: Separate passages into golden (qrel=3) and distractors (qrel=0/1)
# Ensure at least 5 golden + 20 distractors per query

# Step 4: Construct three context variants per query
def build_context(golden_docs, distractor_docs, condition):
    if condition == "A":  # Primacy
        return golden_docs + distractor_docs
    elif condition == "B":  # Middle
        half = len(distractor_docs) // 2
        return distractor_docs[:half] + golden_docs + distractor_docs[half:]
    elif condition == "C":  # Recency
        return distractor_docs + golden_docs
```

### Phase 2: Generation (Week 2–3)

```python
# Run Llama 3.1 8B and Qwen 2.5 7B via vLLM (same as LONGINOUTBENCH setup)
# Temperature: 0.3 (matching LONGINOUTBENCH writing temperature)
# Max new tokens: 600 (400-word answers at ~1.5 tokens/word)
# Batch all conditions per query to save I/O overhead
```

### Phase 3: Nugget Generation (Week 3)

```python
# For each query, extract nuggets offline from golden documents using GPT-4o
# Target: 6–10 nuggets per query
# Validate nuggets manually for a 20-query sample to check quality
# Total API calls: ~120 queries × 1 call = ~120 GPT-4o calls (cheap)
```

### Phase 4: Evaluation (Week 4)

```python
# For each (query, condition, model) triple:
#   - Evaluate Comprehensiveness: score each nugget against the answer
#   - Evaluate Support: score each answer sentence against the documents
# Total API calls: ~720 answers × ~8 nuggets × 1 = ~5,760 GPT-4o nugget calls
#                + ~720 answers × ~12 sentences × 1 = ~8,640 GPT-4o support calls
# Use GPT-4o-mini for support (sentence-level task is simpler; matches TREC cost reduction)
```

### Phase 5: Analysis (Week 5)

```python
import scipy.stats

# 1. Plot U-curve: mean C_score and S_score per condition (A/B/C) per model
# 2. Statistical test: Wilcoxon signed-rank test (A vs B), (C vs B)
# 3. Citation bias analysis: chi-square test on position-weighted citation frequency
# 4. Failure mode categorization: classify Condition B answers as
#    hallucination / omission / distractor-copying
```

---

## 9. Tools and Infrastructure

| Component | Tool |
|---|---|
| Retrieval | Pyserini (BM25 over MS MARCO V2.1) |
| Generation | vLLM (Llama 3.1 8B + Qwen 2.5 7B on A100 40GB) |
| Embedding | bge-base-en-v1.5 (for optional RAL-WRITER extension) |
| Evaluation | GPT-4o + GPT-4o-mini via OpenAI API |
| Nugget framework | AutoNuggetizer (Lin et al., 2024) |
| Statistics | SciPy, Pandas, Matplotlib |
| TREC data tools | ir_datasets, Anserini, Ragnarök |

---

## 10. Academic Contribution

This study makes the following novel contributions:

1. **First positional bias study on real-world non-factoid RAG**: All prior LitM studies use factoid QA or synthetic tasks. TREC RAG 2024 provides real Bing queries with human-curated relevance judgments.

2. **Validation of LONGINOUTBENCH findings in the wild**: Directly tests whether the LitM degradation measured synthetically transfers to messy real-world documents, strengthening the external validity of Zhang et al. (2025).

3. **Support score as a hallucination detector**: When the model "loses" middle-positioned golden evidence, does it fabricate substitutes? This provides a novel mechanistic link between positional bias and hallucination in long-form generation.

4. **Practical blueprint for RAG pipeline design**: Results directly inform how retrieval re-ranking systems should order passages before generation — a directly actionable finding for enterprise RAG architects.

---

## 11. References

- Zhang et al. (2025). *Lost-in-the-Middle in Long-Text Generation: Synthetic Dataset, Evaluation Framework, and Mitigation.* arXiv:2503.06868
- Liu et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 12:157–173
- Pradeep et al. (2024). *Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 RAG Track.* ECIR 2025
- Thakur et al. (2025). *Support Evaluation for the TREC 2024 RAG Track: Comparing Human vs. LLM Judges.* arXiv:2504.15205
- Lin et al. (2024). *Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework.* arXiv:2411.09607
- Bai et al. (2024). *LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs.* arXiv:2408.07055