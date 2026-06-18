"""
03_nugget_doc_alignment.py

Aligns extracted nuggets to the query's golden passages using SentenceTransformers
(BAAI/bge-base-en-v1.5) and cosine similarity. Computes nugget concentration
per golden document and ranks them descending.
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Force UTF-8 encoding for stdout and stderr (essential on Windows)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("03_nugget_doc_alignment")

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"

def save_atomic(data: dict, filepath: Path):
    """Saves data to a JSON file atomically using a temp file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(filepath)

def main():
    nuggets_path = DATA_PROCESSED / "nuggets.json"
    passages_path = DATA_PROCESSED / "per_query_passages.json"
    output_path = DATA_PROCESSED / "nugget_doc_alignment.json"

    if not nuggets_path.exists():
        logger.error(f"Nuggets file not found at {nuggets_path}. Run 02_nugget_creation.py first.")
        sys.exit(1)
    if not passages_path.exists():
        logger.error(f"Passages file not found at {passages_path}. Run 01_data_preparation.py first.")
        sys.exit(1)

    # Load data
    logger.info("Loading inputs...")
    with open(nuggets_path, "r", encoding="utf-8") as f:
        nuggets_data = json.load(f)
    with open(passages_path, "r", encoding="utf-8") as f:
        per_query_passages = json.load(f)

    # Initialize SentenceTransformer model (downloads on first run)
    logger.info("Initializing SentenceTransformer Model (BAAI/bge-base-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    alignment_results = {}
    all_similarities = []
    low_sim_warnings = 0

    logger.info(f"Starting alignment for {len(nuggets_data)} queries...")
    for idx, (query_id, nugget_info) in enumerate(nuggets_data.items(), start=1):
        query_text = nugget_info["query_text"]
        all_nuggets = nugget_info.get("all_nuggets", [])
        
        if not all_nuggets:
            logger.warning(f"No nuggets found for query {query_id}. Skipping.")
            continue

        # Get golden passages for this query
        query_passage_data = per_query_passages.get(query_id, {})
        golden_passages = query_passage_data.get("golden_passages", [])
        
        if not golden_passages:
            logger.warning(f"No golden passages found for query {query_id}. Skipping.")
            continue

        # Extract texts
        nugget_texts = [n["text"] for n in all_nuggets]
        passage_texts = [p["text"] for p in golden_passages]
        passage_ids = [p["docid"] for p in golden_passages]
        bm25_map = {p["docid"]: p.get("bm25_score", 0.0) for p in golden_passages}

        # Encode texts
        nugget_embeddings = model.encode(nugget_texts, normalize_embeddings=True)
        passage_embeddings = model.encode(passage_texts, normalize_embeddings=True)

        # Compute cosine similarity (dot product since normalized)
        sim_matrix = np.dot(nugget_embeddings, passage_embeddings.T)

        nugget_sources = {}
        doc_nugget_counts = {pid: 0 for pid in passage_ids}

        # Align each nugget to the passage with the highest similarity
        for n_idx, nugget in enumerate(all_nuggets):
            n_id = str(nugget["id"])
            best_p_idx = int(np.argmax(sim_matrix[n_idx]))
            best_sim = float(sim_matrix[n_idx][best_p_idx])
            best_docid = passage_ids[best_p_idx]

            nugget_sources[n_id] = {
                "docid": best_docid,
                "similarity": round(best_sim, 4)
            }
            doc_nugget_counts[best_docid] += 1
            all_similarities.append(best_sim)

            if best_sim < 0.5:
                low_sim_warnings += 1
                logger.warning(f"Low similarity match for query {query_id}, nugget {n_id}: sim={best_sim:.4f}")

        # Rank golden docs: primary key nugget count (descending), secondary key BM25 score (descending)
        ranked_golden_docs = sorted(
            passage_ids,
            key=lambda pid: (doc_nugget_counts[pid], bm25_map.get(pid, 0.0)),
            reverse=True
        )

        alignment_results[query_id] = {
            "doc_nugget_counts": doc_nugget_counts,
            "nugget_sources": nugget_sources,
            "golden_docs_ranked_by_concentration": ranked_golden_docs
        }

        # Logging concentration distribution
        counts_list = list(doc_nugget_counts.values())
        logger.info(
            f"[{idx}/{len(nuggets_data)}] Query {query_id}: {len(all_nuggets)} nuggets, "
            f"{len(golden_passages)} golden docs. Concentration: Max={max(counts_list)}, Mean={np.mean(counts_list):.2f}"
        )

    # Save outputs
    logger.info(f"Saving final alignment results to {output_path}...")
    save_atomic(alignment_results, output_path)

    # Overall metrics
    mean_sim = np.mean(all_similarities) if all_similarities else 0.0
    logger.info("══════════════════════════════════════")
    logger.info("ALIGNMENT COMPLETE")
    logger.info("══════════════════════════════════════")
    logger.info(f"Total nuggets aligned:    {len(all_similarities)}")
    logger.info(f"Mean similarity score:    {mean_sim:.4f}")
    logger.info(f"Nuggets with sim < 0.5:   {low_sim_warnings}")
    logger.info("══════════════════════════════════════")

if __name__ == "__main__":
    main()
