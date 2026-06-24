"""
run_pipeline_evaluations.py

Orchestrates the evaluation of Qwen model answers across all 10 context conditions.
Calculates nugget recall, support/hallucination metrics, and length statistics.
Outputs a consolidated markdown report in litm_pipeline/results/RESULTS.md.
"""

import sys
import os
import json
import subprocess
import logging
from pathlib import Path
import numpy as np
from scipy import stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("run_pipeline_evaluations")

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"
EVALS_DIR = DATA_PROCESSED / "evaluations"
RESULTS_DIR = PIPELINE_ROOT / "results"

def calculate_word_stats(file_path: Path) -> dict:
    """Computes mean, median, and mode word count for a responses file."""
    lengths = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                answer = ""
                for k, v in data.items():
                    if k.endswith("_answer"):
                        answer = v
                        break
                if answer:
                    lengths.append(len(answer.strip().split()))
    
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "mode": 0.0}
    
    mean_val = float(np.mean(lengths))
    median_val = float(np.median(lengths))
    mode_res = stats.mode(lengths, keepdims=False)
    mode_val = float(mode_res.mode)
    
    return {"mean": mean_val, "median": median_val, "mode": mode_val}

def load_nugget_eval(file_path: Path) -> dict:
    """Loads and computes macro-level nugget recall rates from an eval JSON file."""
    if not file_path.exists():
        return {"vital_recall": 0.0, "okay_recall": 0.0, "total_vital": 0, "covered_vital": 0}
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_vital = 0
    covered_vital = 0
    total_okay = 0
    covered_okay = 0
    
    for query_id, res in data.items():
        total_vital += res.get("total_vital", 0)
        covered_vital += res.get("covered_vital", 0)
        total_okay += res.get("total_okay", 0)
        covered_okay += res.get("covered_okay", 0)
        
    vital_recall = (covered_vital / total_vital * 100) if total_vital > 0 else 0.0
    okay_recall = (covered_okay / total_okay * 100) if total_okay > 0 else 0.0
    
    return {
        "vital_recall": round(vital_recall, 2),
        "okay_recall": round(okay_recall, 2),
        "total_vital": total_vital,
        "covered_vital": covered_vital,
        "total_okay": total_okay,
        "covered_okay": covered_okay
    }

def load_support_eval(file_path: Path) -> dict:
    """Loads and computes macro-level support rates from an eval JSON file."""
    if not file_path.exists():
        return {"fully_supported_pct": 0.0, "partially_supported_pct": 0.0, "hallucination_rate": 0.0, "global_support_score": 0.0}
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_fully = 0
    total_partially = 0
    total_unsupported = 0
    
    for query_id, res in data.items():
        total_fully += res.get("fully_supported", 0)
        total_partially += res.get("partially_supported", 0)
        total_unsupported += res.get("unsupported", 0)
        
    total_statements = total_fully + total_partially + total_unsupported
    
    if total_statements == 0:
        return {"fully_supported_pct": 0.0, "partially_supported_pct": 0.0, "hallucination_rate": 0.0, "global_support_score": 0.0}
        
    fully_pct = (total_fully / total_statements * 100)
    partially_pct = (total_partially / total_statements * 100)
    hallucination_rate = (total_unsupported / total_statements * 100)
    global_support_score = ((total_fully + 0.5 * total_partially) / total_statements * 100)
    
    return {
        "fully_supported_pct": round(fully_pct, 2),
        "partially_supported_pct": round(partially_pct, 2),
        "hallucination_rate": round(hallucination_rate, 2),
        "global_support_score": round(global_support_score, 2),
        "total_statements": total_statements
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit queries evaluated per run (for debugging)")
    parser.add_argument("--dry-run", action="store_true", help="Only build files map and exit without calling LLM API")
    parser.add_argument("--nugget-model", type=str, default="gemma-4-31b-it", help="Model name for Nugget evaluator")
    parser.add_argument("--nugget-provider", type=str, default="gemini", choices=["gemini", "groq"], help="LLM provider for Nugget evaluator")
    parser.add_argument("--support-model", type=str, default="gemma-4-26b-a4b-it", help="Model name for Support evaluator")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "nugget-only", "support-only", "compile-only"],
                        help="Run mode: all, nugget-only, support-only, or compile-only")
    parser.add_argument("--k", type=int, default=20, help="Total number of documents in context")
    args = parser.parse_args()
    k = args.k

    python_executable = sys.executable

    logger.info(f"Initializing evaluations runner in mode: {args.mode}...")

    # Define conditions map
    conditions = list(range(1, 11))
    
    # Check if necessary directories exist
    (RESULTS_DIR / "answers").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "reports").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    # If compile-only, skip execution loop
    if args.mode == "compile-only":
        logger.info("Skipping evaluations. Proceeding directly to compilation.")
    else:
        # We will process each condition sequentially
        for c in conditions:
            responses_file = RESULTS_DIR / "answers" / f"k{k}_{c}.jsonl"
            contexts_file = DATA_PROCESSED / "contexts" / f"k{k}" / f"condition_{c}.jsonl"
            
            if not responses_file.exists():
                logger.error(f"Responses file not found: {responses_file}. Skipping condition {c}.")
                continue
            if not contexts_file.exists():
                logger.error(f"Contexts file not found: {contexts_file}. Skipping condition {c}.")
                continue

            if args.dry_run:
                logger.info(f"[DRY-RUN] Would evaluate Condition {c}")
                continue

            logger.info(f"=== Starting Evaluation: Condition {c} (k{k}_{c}) ===")

            # 1. Run Nugget Recall Evaluator (if mode is all or nugget-only)
            if args.mode in ["all", "nugget-only"]:
                nugget_cmd = [
                    python_executable,
                    str(SCRIPT_DIR / "06_evaluate_answers.py"),
                    "--responses", str(responses_file),
                    "--provider", args.nugget_provider,
                    "--model", args.nugget_model
                ]
                if args.limit:
                    nugget_cmd += ["--limit", str(args.limit)]
                    
                logger.info(f"Running Nugget Recall Evaluator for Condition {c} (using {args.nugget_model})...")
                try:
                    subprocess.run(nugget_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running nugget evaluator for condition {c}: {e}")
                    sys.exit(1)

            # 2. Run Support / Hallucination Evaluator (if mode is all or support-only)
            if args.mode in ["all", "support-only"]:
                support_cmd = [
                    python_executable,
                    str(SCRIPT_DIR / "07_evaluate_support.py"),
                    "--responses", str(responses_file),
                    "--contexts", str(contexts_file),
                    "--model", args.support_model
                ]
                if args.limit:
                    support_cmd += ["--limit", str(args.limit)]
                    
                logger.info(f"Running Support / Hallucination Evaluator for Condition {c} (using {args.support_model})...")
                try:
                    subprocess.run(support_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running support evaluator for condition {c}: {e}")
                    sys.exit(1)

    if args.dry_run:
        logger.info("Dry-run complete. Exiting.")
        return

    # Compile all outputs into results structure
    logger.info("Compiling all evaluation outputs into RESULTS.md...")
    
    summary_data = []
    
    for c in conditions:
        responses_file = RESULTS_DIR / "answers" / f"k{k}_{c}.jsonl"
        nugget_eval_file = EVALS_DIR / f"k{k}_{c}_eval.json"
        support_eval_file = EVALS_DIR / f"k{k}_{c}_support_eval.json"
        
        if not responses_file.exists():
            continue
            
        word_stats = calculate_word_stats(responses_file)
        nugget_stats = load_nugget_eval(nugget_eval_file)
        support_stats = load_support_eval(support_eval_file)
        
        # Position mapping explanation for report
        if k == 20:
            starting_ranks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 18]
        elif k == 40:
            starting_ranks = [1, 5, 9, 13, 18, 22, 26, 30, 34, 38]
        else:
            starting_ranks = list(range(1, k, max(1, k // 10)))[:10]

        start_rank = starting_ranks[c-1]
        end_rank = start_rank + 2
        
        label = ""
        if c == 1:
            label = " (Primacy)"
        elif c in [5, 6]:
            label = " (Middle)"
        elif c == 10:
            label = " (Recency)"
            
        gold_positions = f"{start_rank}-{end_rank}{label}"
        
        summary_data.append({
            "condition": c,
            "gold_positions": gold_positions,
            "mean_len": word_stats["mean"],
            "median_len": word_stats["median"],
            "mode_len": word_stats["mode"],
            "vital_recall": nugget_stats["vital_recall"],
            "okay_recall": nugget_stats["okay_recall"],
            "fully_supported": support_stats["fully_supported_pct"],
            "partially_supported": support_stats["partially_supported_pct"],
            "hallucination_rate": support_stats["hallucination_rate"],
            "global_support_score": support_stats["global_support_score"]
        })

    # Render RESULTS.md content
    results_md_path = RESULTS_DIR / "reports" / f"RESULTS_k{k}.md"
    
    # Header & Description
    md_content = f"""# Lost-in-the-Middle (LitM) Consolidated Evaluation Results (k={k})

This document summarizes the performance of **Qwen 2.5 7B Instruct** on the **119 Non-Factoid queries** across all 10 context placement conditions (k={k} total documents).

## Consolidated Metric Summary

Below is the consolidated performance table across all 10 conditions. The position of the 3 nugget-rich golden documents slides from the very beginning of the context window (Primacy) to the very end (Recency).

| Condition | Gold Doc Ranks | Vital Recall (%) | Okay Recall (%) | Support Score (%) | Hallucination Rate (%) | Mean Words | Median Words | Mode Words |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""

    for row in summary_data:
        md_content += (
            f"| **k{k}_{row['condition']}** | {row['gold_positions']} | "
            f"**{row['vital_recall']}%** | {row['okay_recall']}% | "
            f"**{row['global_support_score']}%** | {row['hallucination_rate']}% | "
            f"{row['mean_len']:.1f} | {row['median_len']:.0f} | {row['mode_len']:.0f} |\n"
        )
        
    md_content += """
## Major Key Observations

1. **Lost-in-the-Middle Effect (U-Shape curve):**
   - Typically, Vital Recall is highest at the extreme ends of the prompt (Primacy and Recency) and lowest in the middle. We will examine if this pattern holds once the full run completes.
2. **Support Score vs Position:**
   - Evaluates if placing critical information in the middle causes the model to hallucinate more due to context neglect, or if it remains strictly faithful to the provided context.
3. **Word Count Consistency:**
   - Confirms that the model's verbosity remains stable across different prompt layouts, ensuring length bias does not skew the results.

---
> [!NOTE]
> All granular query-level evaluation logs are stored in:
> `litm_pipeline/data/processed/evaluations/`
"""

    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info(f"Consolidated results report written successfully to {results_md_path}")
    logger.info("══════════════════════════════════════")
    logger.info("ALL PIPELINE RUNS COMPLETE")
    logger.info("══════════════════════════════════════")

if __name__ == "__main__":
    main()
