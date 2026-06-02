import os
import sys
import json
import gzip
import string
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ollama

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("live_output_llama3.txt")

MODEL_NAME = "llama3.1:8b"

datasets = {
    "10_total_documents": [0, 4, 9],
}
NUM_QUESTIONS_TO_TEST = 100

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

print(f"Testing {NUM_QUESTIONS_TO_TEST} Questions per position on local Ollama using {MODEL_NAME}...")
print("=" * 60)

results_data = []
total_start = time.time()

for folder, positions in datasets.items():
    print(f"\n{'='*40}\n>>> TESTING DATASET: {folder} <<<\n{'='*40}")
    base_path = f"lost-in-the-middle/qa_data/{folder}"
    context_size = folder.split("_")[0]

    for pos in positions:
        filename = f"nq-open-{folder}_gold_at_{pos}.jsonl.gz"
        filepath = os.path.join(base_path, filename)

        correct_matches = 0
        pos_start = time.time()

        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            print(f"\n--- Testing Position {pos} ---")

            for q_idx in range(NUM_QUESTIONS_TO_TEST):
                line = f.readline()
                if not line:
                    break
                data = json.loads(line)

                target_query = data["question"]
                target_answer = data["answers"][0]
                ctxs = data["ctxs"]

                context_parts = []
                for i, ctx in enumerate(ctxs):
                    context_parts.append(f"Document [{i+1}](Title: {ctx['title']}) {ctx['text']}")
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
                            'num_ctx': 2048,
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

                    print(f"\n[Question {q_idx+1}] ({q_time:.1f}s)")
                    print(f"Target Answer: {target_answer}")
                    print(f"Model Response: {output.strip()}")
                    print(f"Match: {success}")

                except Exception as e:
                    print(f"\nError connecting to Ollama: {str(e)}")
                    print("Make sure you have Ollama installed and running on your computer!")
                    break

        pos_time = time.time() - pos_start
        accuracy = (correct_matches / NUM_QUESTIONS_TO_TEST) * 100
        print(f"\n>> Position {pos} Accuracy: {accuracy:.1f}% ({correct_matches}/{NUM_QUESTIONS_TO_TEST}) [{pos_time:.0f}s]")

        results_data.append({
            "Context Size (Documents)": context_size,
            "Position of Answer (Gold Index)": pos,
            "Accuracy (%)": accuracy
        })

total_time = time.time() - total_start
print(f"\nTotal time: {total_time/60:.1f} minutes")

print("\nGenerating Seaborn Graph...")
df = pd.DataFrame(results_data)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

plot = sns.lineplot(
    data=df,
    x="Position of Answer (Gold Index)",
    y="Accuracy (%)",
    hue="Context Size (Documents)",
    marker="o",
    palette="deep"
)

plt.title(f"Lost in the Middle Evaluation (Ollama - {MODEL_NAME})", fontsize=14)
plt.ylim(0, 100)
plt.xticks(df["Position of Answer (Gold Index)"].unique())

output_image = "ollama_graph_llama3.png"
plt.tight_layout()
plt.savefig(output_image, dpi=300)

print(f"DONE! The graph has been saved to: {output_image}")
