# Lost in the Middle — LLM Evaluation

> A replication and extension of Liu et al. (2023), testing whether modern LLMs still struggle to retrieve information from the middle of long contexts.

**Paper:** [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu, Lin, Hewitt et al., 2023

---

## Overview

This project evaluates the **"Lost in the Middle"** phenomenon across three locally runnable / API-accessible models using the Natural Questions dataset. We test whether positioning a relevant document in the middle of a long context degrades model accuracy — and compare our findings against the original paper's results.

---

## Models Evaluated

| # | Model | Provider | Access | Context Window | Scale |
|---|-------|----------|--------|----------------|-------|
| 1 | **Llama 3.1 8B** | Meta | Local (Ollama) | 128K tokens | Full dataset: 2,655 q/pos |
| 2 | **Phi-3 Mini** | Microsoft | Local (Ollama) | 128K tokens | 300 q/pos |
| 3 | **Gemma 4 31B-IT** | Google | Gemini API | 1M tokens | ~100 q/pos |
| — | *Llama-2-70b-chat (paper)* | *Meta* | *VLLM* | *4K tokens* | *2,655 q/pos* |

---

## Key Results

### 10-Document Context

| Model | Pos 0 (Beginning) | Pos 4 (Middle) | Pos 9 (End) | Mid Drop | U-Shape? |
|-------|:-----------------:|:--------------:|:-----------:|:--------:|:--------:|
| Llama 3.1 8B | 49.49% | 45.69% | 44.90% | −3.80% | Partial |
| Phi-3 Mini | 38.33% | 31.00% | 38.67% | −7.33% | ✅ Strong |
| Gemma 4 31B-IT *(truncated)* | 53.93% | 56.12% | 55.56% | flat | ❌ Flat |
| *Llama-2-70b-chat (paper)* | *~64%* | *~52%* | *~61%* | *~−12%* | *✅ Classic* |

See [`experiment_results.md`](./experiment_results.md) for the full analysis including 20-doc and 30-doc results.

---

## Project Structure

```
.
├── src/                                   # Evaluation scripts
│   ├── test.py                            # Gemma 4 31B-IT via Gemini API (10 docs)
│   ├── test_gemma_baseline.py             # Gemma 4 closed-book baseline (Gemini API)
│   ├── test_ollama.py                     # Phi-3 Mini via Ollama (10 docs, 300 q/pos)
│   ├── test_baseline.py                   # Llama 3.1 8B closed-book baseline (Ollama)
│   ├── check_context.py                   # Utility: measure prompt token lengths
│   ├── kaggle_llama3.py                   # Llama 3.1 8B reference script (10 docs)
│   ├── lost_in_the_middle_llama3_10docs.ipynb  # Kaggle notebook: Llama3 10 docs
│   ├── lost_in_the_middle_llama3_20docs.ipynb  # Kaggle notebook: Llama3 20 docs
│   └── lost_in_the_middle_llama3_30docs.ipynb  # Kaggle notebook: Llama3 30 docs
│
├── utilities/                             # Post-processing & graphing utilities
│   ├── parse_logs.py                      # Parse live_output.txt logs into JSON
│   ├── plot_gemma_graph.py                # Plot Gemma 4 evaluation results
│   ├── plot_phi3_zoomed.py                # Plot Phi-3 zoomed results (no baseline)
│   ├── plot_llama3_comparison.py          # Plot Llama3 multi-context comparison
│   ├── plot_baseline_graphs.py            # Overlay baseline on evaluation graphs
│   ├── plot_zoomed_no_baseline.py         # Zoomed Llama3 10-doc graph
│   ├── plot_paper_10_30_docs.py           # Reproduce paper figures (10 & 30 docs)
│   ├── plot_paper_results.py              # Reproduce paper figures (20 docs)
│   └── recalculate_gemma_accuracy.py      # Recalculate Gemma accuracy with truncation
│
├── results/
│   ├── gemma_results/
│   │   └── gemma_4_31b_it_graph.png       # Gemma evaluation graph (truncated)
│   ├── llama 3.1 8b/
│   │   ├── 10documents_results/           # Results JSON + graphs for 10-doc run
│   │   ├── 20documents_results/           # Results JSON + graphs for 20-doc run
│   │   └── 30documents_results/           # Results JSON + graphs for 30-doc run
│   ├── phi3_results/
│   │   └── ollama_graph_zoomed.png        # Phi-3 U-curve graph (zoomed, no baseline)
│   ├── paper_original_10docs.png          # Original paper figure (10 docs)
│   ├── paper_original_30docs.png          # Original paper figure (30 docs)
│   └── paper_original_results.png         # Original paper figure (20 docs)
│
├── data/                                  # Dataset files (git-ignored, large)
│   └── qa_data/                           # NQ-open JSONL files per position/context
│
├── lost-in-the-middle/                    # Original paper repo (git submodule, ignored)
├── experiment_results.md                  # Full experiment write-up & analysis
├── requirements.txt                       # Python dependencies
└── .env                                   # API keys (git-ignored)
```

---

## Setup

### 1. Clone and set up the environment

```bash
git clone https://github.com/your-username/lost-in-the-middle-eval.git
cd lost-in-the-middle-eval

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Only required for the Gemma 4 31B-IT experiments.

### 3. Set up the dataset

The dataset is the Natural Questions subset from the original paper's repository. Place the unzipped `qa_data/` folder under `data/`:

```
data/
└── qa_data/
    ├── 10_total_documents/
    │   ├── nq-open-10_total_documents_gold_at_0.jsonl
    │   ├── nq-open-10_total_documents_gold_at_4.jsonl
    │   └── nq-open-10_total_documents_gold_at_9.jsonl
    ├── 20_total_documents/
    └── 30_total_documents/
```

> The original `.jsonl.gz` files from the paper can also be used — `check_context.py` and the Gemma/Phi-3 scripts support gzip. The Kaggle notebooks use unzipped `.jsonl` files.

---

## Running the Experiments

### Phi-3 Mini (Ollama, local)

Make sure Ollama is running and `phi3` model is pulled:

```bash
ollama pull phi3
python src/test_ollama.py
```

Outputs to: `results/phi3_results/`

### Gemma 4 31B-IT (Gemini API)

```bash
python src/test.py
```

Outputs to: `results/gemma_results/`

### Llama 3.1 8B (Kaggle GPU)

The Llama experiments require a Kaggle GPU environment (T4 x2) due to runtime length (~4–28 hours). Use the notebooks in `src/`:

- `src/lost_in_the_middle_llama3_10docs.ipynb` — 10-document evaluation
- `src/lost_in_the_middle_llama3_20docs.ipynb` — 20-document evaluation  
- `src/lost_in_the_middle_llama3_30docs.ipynb` — 30-document evaluation

Upload the dataset to Kaggle and update `DATASET_PATH` in the notebook config before running.

### Baselines (no-document / closed-book)

```bash
# Llama 3.1 8B closed-book baseline
python src/test_baseline.py

# Gemma 4 closed-book baseline
python src/test_gemma_baseline.py
```

---

## Generating Graphs

All graph scripts are in `utilities/`. Run from the project root:

```bash
# Phi-3 zoomed U-curve
python utilities/plot_phi3_zoomed.py

# Gemma 4 evaluation graph (with truncation annotation)
python utilities/plot_gemma_graph.py

# Llama3 10-doc zoomed graph (no baseline)
python utilities/plot_zoomed_no_baseline.py

# Llama3 comparison across 10/20/30 docs
python utilities/plot_llama3_comparison.py
```

---

## Methodology Notes

- **Answer matching:** Case-insensitive substring match on `answers[0]` (the primary answer)
- **Response truncation:** Applied for Gemma (verbose model) — response truncated at first `\n` before matching, consistent with the original paper's `best_subspan_em`
- **Context window:** `num_ctx=4096` for Phi-3 and Llama3 10-doc; `num_ctx=8192` for Llama3 30-doc
- **Temperature:** `0.0` for all models (deterministic)

---

## Citation

```bibtex
@article{liu2023lost,
  title={Lost in the Middle: How Language Models Use Long Contexts},
  author={Liu, Nelson F and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={arXiv preprint arXiv:2307.03172},
  year={2023}
}
```
