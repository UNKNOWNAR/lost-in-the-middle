"""
Lost in the Middle - Kaggle Notebook (Save Version)
Model: llama3.1:8b | All 2,655 questions | 10 docs
GPU: 2x T4 | Estimated time: ~9-11 hours

Usage: Paste each cell into a Kaggle notebook, select GPU T4 x2, then Save Version (Save & Run All).
Make sure Internet is ON in notebook settings.
Add the dataset: upload the lost-in-the-middle repo as a Kaggle dataset,
and update DATASET_PATH below to match your dataset path (e.g., /kaggle/input/your-dataset-name).
"""

# ============================================================
# CELL 1: Install dependencies, Ollama and pull model
# ============================================================

import subprocess
import time

print("Installing dependencies...")
subprocess.run("apt-get update && apt-get install -y zstd", shell=True, check=True)

print("Installing Ollama...")
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

print("Starting server...")
subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(10)

print("Pulling model...")
subprocess.run("ollama pull llama3.1:8b", shell=True, check=True)
print("Ollama ready!")

# ============================================================
# CELL 2: Run the experiment
# ============================================================

import os
import json
import gzip
import string
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ollama

# --- CONFIG ---
MODEL_NAME = "llama3.1:8b"
NUM_CTX = 4096
DATASET_PATH = "/kaggle/input/datasets/arinjaysarkar/qa-questions"
OUTPUT_DIR = "/kaggle/working"

datasets = {
    "10_total_documents": [0, 4, 9],
}

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

results_data = []
all_responses = []
total_start = time.time()

print(f"Model: {MODEL_NAME} | Context: {NUM_CTX} tokens")
print("=" * 60)

for folder, positions in datasets.items():
    base_path = os.path.join(DATASET_PATH, "qa_data", folder)
    context_size = folder.split("_")[0]

    for pos in positions:
        filename = f"nq-open-{folder}_gold_at_{pos}.jsonl"
        filepath = os.path.join(base_path, filename)

        correct_matches = 0
        total_questions = 0
        pos_start = time.time()

        print(f"\n{'='*40}")
        print(f">>> {folder} | Position {pos} <<<")
        print(f"{'='*40}")

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            total_questions = len(lines)

            for q_idx, line in enumerate(lines):
                data = json.loads(line)

                target_query = data["question"]
                target_answer = data["answers"][0]
                ctxs = data["ctxs"]

                context_parts = []
                for j, ctx in enumerate(ctxs):
                    context_parts.append(f"Document [{j+1}](Title: {ctx['title']}) {ctx['text']}")
                context_str = "\n".join(context_parts)

                prompt = (
                    "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n"
                    f"{context_str}\n\n"
                    f"Question: {target_query}\n"
                    "Answer: (Keep the answer as concise as possible, preferably one or two words. Do not write full sentences.)"
                )

                try:
                    q_start = time.time()
                    response = ollama.chat(
                        model=MODEL_NAME,
                        messages=[{'role': 'user', 'content': prompt}],
                        options={
                            'num_ctx': NUM_CTX,
                            'temperature': 0.0
                        }
                    )
                    q_time = time.time() - q_start
                    output = response['message']['content']

                    normalized_prediction = normalize_answer(output)
                    normalized_ground_truth = normalize_answer(target_answer)

                    success = normalized_ground_truth in normalized_prediction
                    if success:
                        correct_matches += 1

                    all_responses.append({
                        "folder": folder,
                        "position": pos,
                        "q_idx": q_idx,
                        "question": target_query,
                        "target_answer": target_answer,
                        "model_response": output.strip(),
                        "match": success,
                        "time_sec": round(q_time, 2)
                    })

                    if (q_idx + 1) % 50 == 0 or q_idx < 3:
                        accuracy_so_far = (correct_matches / (q_idx + 1)) * 100
                        elapsed = time.time() - pos_start
                        eta = (elapsed / (q_idx + 1)) * (total_questions - q_idx - 1)
                        print(f"[Q{q_idx+1}/{total_questions}] ({q_time:.1f}s) "
                              f"Acc: {accuracy_so_far:.1f}% | "
                              f"ETA: {eta/60:.0f}min | "
                              f"{target_answer} -> {output.strip()[:50]} | {success}")

                except Exception as e:
                    print(f"Error at Q{q_idx+1}: {str(e)}")
                    all_responses.append({
                        "folder": folder,
                        "position": pos,
                        "q_idx": q_idx,
                        "question": target_query,
                        "target_answer": target_answer,
                        "model_response": f"ERROR: {str(e)}",
                        "match": False,
                        "time_sec": 0
                    })

        accuracy = (correct_matches / total_questions) * 100
        pos_time = time.time() - pos_start
        print(f"\n>> Position {pos} DONE: {accuracy:.1f}% ({correct_matches}/{total_questions}) [{pos_time/60:.1f}min]")

        results_data.append({
            "Context Size (Documents)": context_size,
            "Position of Answer (Gold Index)": pos,
            "Accuracy (%)": accuracy,
            "Correct": correct_matches,
            "Total": total_questions
        })

        # Save intermediate results after each position
        with open(os.path.join(OUTPUT_DIR, "results_llama3.json"), "w") as f:
            json.dump(results_data, f, indent=2)
        with open(os.path.join(OUTPUT_DIR, "all_responses_llama3.json"), "w") as f:
            json.dump(all_responses, f, indent=2)

total_time = time.time() - total_start
print(f"\n{'='*60}")
print(f"ALL DONE! Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print(f"{'='*60}")

# ============================================================
# CELL 3: Generate graph
# ============================================================

df = pd.DataFrame(results_data)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="Position of Answer (Gold Index)",
    y="Accuracy (%)",
    hue="Context Size (Documents)",
    marker="o",
    palette="deep"
)

plt.title(f"Lost in the Middle - Full Evaluation ({MODEL_NAME})", fontsize=14)
plt.ylim(0, 100)
plt.xticks(df["Position of Answer (Gold Index)"].unique())

output_image = os.path.join(OUTPUT_DIR, "kaggle_graph_llama3.png")
plt.tight_layout()
plt.savefig(output_image, dpi=300)
plt.show()

print(f"Graph saved to: {output_image}")
print("\nResults summary:")
print(df.to_string(index=False))
