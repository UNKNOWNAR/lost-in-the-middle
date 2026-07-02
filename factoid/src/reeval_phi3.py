"""
Re-evaluate Phi-3 Mini results with ALL possible answers.

The original run only checked answers[0]. This script:
  1. Parses the saved model responses from live_output_ollama.txt
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

# --- PATHS ---
LOG_FILE       = r"results\phi3_results\live_output_ollama.txt"
DATA_DIR       = r"data\qa_data\10_total_documents"
FOLDER         = "10_total_documents"
POSITIONS      = [0, 4, 9]
NUM_QUESTIONS  = 300

# --- NORMALIZATION ---
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

# --- PARSE LOG FILE ---
print("Parsing log file...")
with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

# Split by position blocks
pos_blocks = re.split(r"--- Testing Position \d+ ---", text)[1:]

parsed_responses = {pos: [] for pos in POSITIONS}

for pos, block in zip(POSITIONS, pos_blocks):
    q_blocks = block.split("[Question ")[1:]
    for qb in q_blocks:
        lines = qb.strip().split("\n")
        # lines[0] is like "1]"
        # Next lines might be:
        # Target Answer: ...
        # Model Response: ...
        # Match: ...
        target_answer = ""
        model_response = ""
        for line in lines:
            if line.startswith("Target Answer:"):
                target_answer = line.split("Target Answer:")[1].strip()
            elif line.startswith("Model Response:"):
                model_response = line.split("Model Response:")[1].strip()
        
        parsed_responses[pos].append(model_response)

print(f"  Parsed {sum(len(v) for v in parsed_responses.values())} records from log.\n")

# --- LOAD ALL ANSWERS FROM ORIGINAL DATASET ---
print("Loading original datasets to get all answer synonyms...")
all_answers_map = {}

for pos in POSITIONS:
    filename = f"nq-open-{FOLDER}_gold_at_{pos}.jsonl.gz"
    filepath = os.path.join(DATA_DIR, filename)
    all_answers_map[pos] = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= NUM_QUESTIONS:
                break
            data = json.loads(line)
            all_answers_map[pos].append(data["answers"])
    print(f"  Position {pos}: loaded {len(all_answers_map[pos])} question answer sets")

print()

# --- RE-EVALUATE ---
comparison_rows = []

for pos in POSITIONS:
    responses = parsed_responses[pos]
    answers_list = all_answers_map[pos]
    
    total = len(responses)
    old_correct = 0
    new_correct = 0
    
    for i in range(total):
        model_output = responses[i]
        all_answers = answers_list[i]
        target_answer = all_answers[0] # Original script used answers[0]
        
        # --- OLD scoring ---
        old_norm = normalize_answer(model_output)
        old_hit = normalize_answer(target_answer) in old_norm
        if old_hit:
            old_correct += 1
            
        # --- NEW scoring ---
        truncated = model_output.split("\n")[0].strip()
        new_norm = normalize_answer(truncated)
        new_hit = any(normalize_answer(ans) in new_norm for ans in all_answers)
        if new_hit:
            new_correct += 1

    old_acc = (old_correct / total) * 100
    new_acc = (new_correct / total) * 100
    delta = new_acc - old_acc

    comparison_rows.append({
        "Position": pos,
        "Total": total,
        "Old Accuracy": f"{old_acc:.2f}%",
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
