"""
Re-evaluate and calculate benchmarks for Qwen 2.5 7B evaluation runs.

This script:
  1. Loads raw model responses from a specified Qwen responses JSON file.
  2. Applies first-line truncation: output.split('\n')[0].strip()
  3. Scans and normalizes the prediction and all valid target answers.
  4. Scoring rule: any(normalized_answer in normalized_prediction for answer in target_answers)
  5. Outputs formatted results per position.
"""

import os
import sys
import json
import string
import re
from collections import defaultdict

# --- NORMALIZATION (paper-exact) ---
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def evaluate_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return

    print(f"Loading responses from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        responses = json.load(f)
    print(f"Loaded {len(responses)} records.")

    # Group responses by position
    by_position = defaultdict(list)
    for r in responses:
        # Some schemas might use 'position', others 'Position of Answer (Gold Index)'
        pos = r.get("position", r.get("Position of Answer (Gold Index)"))
        if pos is not None:
            by_position[int(pos)].append(r)

    results_table = []
    
    print("\nCalculating metrics (Paper-exact best_subspan_em with first-line truncation)...")
    for pos in sorted(by_position.keys()):
        pos_responses = by_position[pos]
        total = len(pos_responses)
        correct = 0

        for r in pos_responses:
            # Handle different JSON schema versions
            raw_output = r.get("raw_output", r.get("model_response", ""))
            target_answers = r.get("target_answers", r.get("target_answer", []))
            
            # If target_answers is a string, wrap in a list
            if isinstance(target_answers, str):
                target_answers = [target_answers]

            # Truncate at first newline
            truncated = raw_output.split("\n")[0].strip()
            normalized_pred = normalize_answer(truncated)

            # Check if any ground truth synonym is contained in the normalized prediction
            match = any(
                normalize_answer(ans) in normalized_pred 
                for ans in target_answers
            )
            
            if match:
                correct += 1

        acc = (correct / total) * 100 if total > 0 else 0.0
        results_table.append({
            "Position": pos,
            "Correct": correct,
            "Total": total,
            "Accuracy": f"{acc:.2f}%"
        })

    # Print nicely formatted table
    print("\n" + "=" * 50)
    print(f"{'Position':^10} | {'Correct':^10} | {'Total':^10} | {'Accuracy':^12}")
    print("-" * 50)
    for row in results_table:
        print(f"{row['Position']:^10} | {row['Correct']:^10} | {row['Total']:^10} | {row['Accuracy']:^12}")
    print("=" * 50 + "\n")

def main():
    if len(sys.argv) > 1:
        evaluate_file(sys.argv[1])
    else:
        # List of common default paths to search if no arg is given
        default_paths = [
            r"results\qwen2.5_7b\20documents_results\all_responses_qwen25.json",
            r"results\qwen2.5_7b\10documents_results\all_responses_qwen25_10docs.json",
            r"results\qwen2.5_7b\30documents_results\all_responses_qwen25_30docs.json"
        ]
        
        found = False
        for path in default_paths:
            if os.path.exists(path):
                evaluate_file(path)
                found = True
        
        if not found:
            print("Usage: python src/reeval_qwen25.py <path_to_responses_json>")
            print("No default Qwen responses JSON file found on disk.")

if __name__ == "__main__":
    main()
