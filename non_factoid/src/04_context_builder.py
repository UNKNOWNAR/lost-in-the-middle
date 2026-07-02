"""
04_context_builder.py

Builds context prompts for all query x condition combinations.
We generate 10 conditions by sliding the 3 highest concentration gold documents
across the context window (ranks 1 to 20).
"""

import sys
import json
import logging
import argparse
import math
from pathlib import Path

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
logger = logging.getLogger("04_context_builder")

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"

# We will generate 10 conditions dynamically based on k
def generate_starting_ranks(k: int, num_conditions: int = 10, n_golden: int = 3) -> list[int]:
    max_start_rank = k - n_golden + 1
    if k == 20:
        return [1, 3, 5, 7, 9, 11, 13, 15, 17, 18]
    elif k == 40:
        return [1, 5, 9, 13, 18, 22, 26, 30, 34, 38]
    
    # Generic fallback for other k
    step = (max_start_rank - 1) / max(1, (num_conditions - 1))
    ranks = [int(round(1 + i * step)) for i in range(num_conditions)]
    # Ensure uniqueness and sorted order
    unique_ranks = sorted(list(set(ranks)))
    # If we lost some due to rounding dupes, just append up to max
    while len(unique_ranks) < num_conditions:
        for val in range(1, max_start_rank + 1):
            if val not in unique_ranks:
                unique_ranks.append(val)
                unique_ranks.sort()
                break
    return unique_ranks[:num_conditions]

def get_title_and_text(passage: dict) -> tuple[str, str]:
    """Retrieves or extracts title and clean text from a passage dict."""
    title = passage.get("title")
    text = passage.get("text", "")

    if title:
        return title, text

    # If title is missing, try to extract from the first line of text
    lines = text.split("\n")
    if lines:
        extracted_title = lines[0].strip()
        if len(extracted_title) < 200:
            remaining_text = "\n".join(lines[1:]).strip()
            if remaining_text:
                return extracted_title, remaining_text
            else:
                return extracted_title, text

    return "Unknown Title", text

def main():
    parser = argparse.ArgumentParser(description="Build contexts for Lost-in-the-Middle evaluation.")
    parser.add_argument("--k", type=int, default=20, help="Total number of documents in context (default: 20)")
    parser.add_argument("--n_golden", type=int, default=3, help="Number of golden documents (default: 3)")
    args = parser.parse_args()
    
    k = args.k
    n_golden = args.n_golden
    starting_ranks = generate_starting_ranks(k, n_golden=n_golden)

    passages_path = DATA_PROCESSED / "per_query_passages.json"
    alignment_path = DATA_PROCESSED / "nugget_doc_alignment.json"
    output_dir = DATA_PROCESSED / "contexts" / f"k{k}"

    if not passages_path.exists():
        logger.error(f"Passages file not found at {passages_path}. Run 01_data_preparation.py first.")
        sys.exit(1)
    if not alignment_path.exists():
        logger.error(f"Alignment file not found at {alignment_path}. Run 03_nugget_doc_alignment.py first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    logger.info("Loading inputs...")
    with open(passages_path, "r", encoding="utf-8") as f:
        per_query_passages = json.load(f)
    with open(alignment_path, "r", encoding="utf-8") as f:
        alignment_data = json.load(f)

    # We will write to 10 separate files
    condition_handlers = {}
    for c_idx in range(1, 11):
        file_path = output_dir / f"condition_{c_idx}.jsonl"
        condition_handlers[c_idx] = open(file_path, "w", encoding="utf-8")

    try:
        logger.info("Constructing contexts for 10 conditions...")
        processed_count = 0

        for query_id, align_info in alignment_data.items():
            query_passage_data = per_query_passages.get(query_id)
            if not query_passage_data:
                logger.warning(f"No passages found for query {query_id}. Skipping.")
                continue

            query_text = query_passage_data["query_text"]
            golden_passages = query_passage_data.get("golden_passages", [])
            distractor_passages = query_passage_data.get("distractor_passages", [])

            # Get the gold documents ordered by concentration rank
            golden_by_id = {p["docid"]: p for p in golden_passages}
            ranked_gold_ids = align_info["golden_docs_ranked_by_concentration"]
            gold_docs = [golden_by_id[docid] for docid in ranked_gold_ids if docid in golden_by_id]

            # Select top M golden documents
            top_gold = gold_docs[:n_golden]
            
            # Select (k - n_golden) distractors
            num_distractors = k - len(top_gold)
            distractors = distractor_passages[:num_distractors]

            if len(top_gold) < n_golden:
                # Fallback: if we somehow have fewer golden docs, fill with distractors
                logger.warning(f"Query {query_id} only has {len(top_gold)} gold docs. Using distractors as filler.")
                additional_distractors_needed = n_golden - len(top_gold)
                num_distractors = k - n_golden
                top_gold += distractor_passages[num_distractors:num_distractors + additional_distractors_needed]
                distractors = distractor_passages[:num_distractors]
            
            # Make sure we have exactly k documents in total
            assert len(top_gold) == n_golden, f"Query {query_id} does not have {n_golden} gold documents"
            assert len(distractors) == k - n_golden, f"Query {query_id} does not have {k-n_golden} distractors"

            # Create the 10 conditions
            for c_idx, start_rank in enumerate(starting_ranks, start=1):
                start_idx = start_rank - 1  # 0-indexed
                
                # Slice and insert
                first_part = distractors[:start_idx]
                second_part = distractors[start_idx:]
                
                context_passages = first_part + top_gold + second_part
                assert len(context_passages) == k, f"Query {query_id} Condition {c_idx} does not have {k} passages"

                # Render details
                rendered_docs = []
                golden_doc_positions = []
                golden_doc_ids = []

                for pos, p in enumerate(context_passages, start=1):
                    docid = p["docid"]
                    is_golden = p in top_gold
                    nugget_count = align_info["doc_nugget_counts"].get(docid, 0) if is_golden else 0
                    title, text_content = get_title_and_text(p)

                    rendered_docs.append({
                        "position": pos,
                        "docid": docid,
                        "is_golden": is_golden,
                        "nugget_count": nugget_count,
                        "title": title,
                        "text": text_content
                    })

                    if is_golden:
                        golden_doc_positions.append(pos)
                        golden_doc_ids.append(docid)

                # Construct JSONL line
                output_line = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "k": k,
                    "condition": c_idx,
                    "n_golden": len(top_gold),
                    "n_distractors": len(distractors),
                    "golden_doc_positions": golden_doc_positions,
                    "golden_doc_ids": golden_doc_ids,
                    "context_docs": rendered_docs
                }

                # Write line
                condition_handlers[c_idx].write(json.dumps(output_line, ensure_ascii=False) + "\n")

            processed_count += 1

        logger.info(f"Successfully constructed contexts for {processed_count} queries across 10 conditions.")

    finally:
        # Safely close all files
        for handler in condition_handlers.values():
            handler.close()

    logger.info("══════════════════════════════════════")
    logger.info("CONTEXT CONSTRUCTION COMPLETE")
    logger.info("══════════════════════════════════════")
    logger.info(f"Total queries written: {processed_count}")
    logger.info(f"Conditions generated:  10 (condition_1.jsonl to condition_10.jsonl)")
    logger.info(f"Output Directory:      {output_dir.relative_to(PIPELINE_ROOT)}")
    logger.info("══════════════════════════════════════")

if __name__ == "__main__":
    main()
