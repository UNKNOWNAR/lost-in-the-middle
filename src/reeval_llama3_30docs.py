"""
Re-evaluate Llama 3.1 8B 30-doc results with ALL possible answers.

The original run only checked answers[0]. This script:
  1. Loads the saved model responses from all_responses_llama3.json
  2. Loads the original JSONL.gz dataset files to get all valid answer synonyms
  3. Re-scores using any() over all answers (same as original paper)
  4. Also applies first-line truncation (paper's best_subspan_em design)
  5. Prints a side-by-side comparison: old vs new accuracy per position
"""

import os
import json
import gzip
import string
import re
from collections import defaultdict

# --- PATHS ---
RESPONSES_FILE = r"results\llama 3.1 8b\30documents_results\all_responses_llama3.json"
DATA_DIR       = r"data\qa_data\30_total_documents"
FOLDER         = "30_total_documents"
POSITIONS      = [0, 4, 9, 14, 19, 24, 29]

# --- NORMALIZATION (identical to original notebook) ---
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

# --- LOAD SAVED RESPONSES ---
print("Loading saved responses...")
with open(RESPONSES_FILE, "r", encoding="utf-8") as f:
    responses = json.load(f)
print(f"  Loaded {len(responses)} records\n")

# --- LOAD ALL ANSWERS FROM ORIGINAL DATASET ---
# Build a dict: position -> q_idx -> list of all answers
print("Loading original datasets to get all answer synonyms...")
all_answers_map = {}   # {pos: {q_idx: [ans1, ans2, ...]}}

for pos in POSITIONS:
    filename = f"nq-open-{FOLDER}_gold_at_{pos}.jsonl.gz"
    filepath = os.path.join(DATA_DIR, filename)
    all_answers_map[pos] = {}
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for q_idx, line in enumerate(f):
            data = json.loads(line)
            all_answers_map[pos][q_idx] = data["answers"]
    print(f"  Position {pos}: loaded {len(all_answers_map[pos])} question answer sets")

print()

# --- RE-EVALUATE ---
# Group responses by position
by_position = defaultdict(list)
for r in responses:
    by_position[r["position"]].append(r)

comparison_rows = []

for pos in POSITIONS:
    pos_responses = sorted(by_position[pos], key=lambda x: x["q_idx"])
    total = len(pos_responses)

    if total == 0:
        continue

    old_correct     = 0   # original: single answer, no truncation
    new_correct     = 0   # new: all answers + first-line truncation

    for r in pos_responses:
        q_idx        = r["q_idx"]
        model_output = r["model_response"]

        # --- OLD scoring (replicate original notebook exactly) ---
        old_answer = r["target_answer"]
        old_norm   = normalize_answer(model_output)
        old_hit    = normalize_answer(old_answer) in old_norm
        if old_hit:
            old_correct += 1

        # --- NEW scoring: truncate at first \n, check all answers ---
        truncated    = model_output.split("\n")[0].strip()
        new_norm     = normalize_answer(truncated)
        all_answers  = all_answers_map[pos][q_idx]
        new_hit      = any(normalize_answer(ans) in new_norm for ans in all_answers)
        if new_hit:
            new_correct += 1

    old_acc = (old_correct / total) * 100
    new_acc = (new_correct / total) * 100
    delta   = new_acc - old_acc

    comparison_rows.append({
        "Position": pos,
        "Total": total,
        "Old Correct (answers[0], no trunc)": old_correct,
        "Old Accuracy": f"{old_acc:.2f}%",
        "New Correct (all answers + trunc)": new_correct,
        "New Accuracy": f"{new_acc:.2f}%",
        "Delta": f"{delta:+.2f}%",
    })

# --- PRINT RESULTS ---
print("=" * 75)
print(f"{'Pos':>4} | {'Total':>6} | {'Old Acc':>10} | {'New Acc':>10} | {'Delta':>8}")
print("-" * 75)
for row in comparison_rows:
    print(f"  {row['Position']:>2} | {row['Total']:>6} | "
          f"{row['Old Accuracy']:>10} | {row['New Accuracy']:>10} | {row['Delta']:>8}")
print("=" * 75)

print("\nChanges applied in new evaluation:")
print("  1. First-line truncation  : output.split('\\n')[0].strip()")
print("  2. All answer synonyms    : any(normalize(ans) in prediction for ans in answers)")
print("\nThese match the original paper's best_subspan_em metric exactly.")
