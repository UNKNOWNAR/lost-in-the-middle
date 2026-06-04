import os
import sys
import time
import json
import gzip
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Force write to disk instantly

    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.makedirs("results/gemma_results", exist_ok=True)
sys.stdout = Logger("results/gemma_results/live_output_gemini_gemma.txt")

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemma-4-31b-it"

datasets = {
    "10_total_documents": [0, 4, 9],
}
NUM_QUESTIONS_TO_TEST = 100
total_api_requests = 0

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

print(f"Testing {NUM_QUESTIONS_TO_TEST} Questions per position and plotting graph...\n" + "="*60)

results_data = []

for folder, positions in datasets.items():
    print(f"\n{'='*40}\n>>> TESTING DATASET: {folder} <<<\n{'='*40}")
    base_path = f"lost-in-the-middle/qa_data/{folder}"
    context_size = folder.split("_")[0] 
    
    for pos in positions:
        filename = f"nq-open-{folder}_gold_at_{pos}.jsonl.gz"
        filepath = os.path.join(base_path, filename)
        
        correct_matches = 0
        
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
                
                prompt = f"""Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).

{context_str}

Question: {target_query}
Answer:"""

                while True:
                    try:
                        total_api_requests += 1
                        request_start_time = time.time()
                        
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                            )
                        )
                        output = response.text
                        
                        normalized_prediction = normalize_answer(output)
                        normalized_ground_truth = normalize_answer(target_answer)
                        
                        success = normalized_ground_truth in normalized_prediction
                        if success:
                            correct_matches += 1
                        
                        # Print detailed results just like before
                        print(f"\n[Question {q_idx+1}]")
                        print(f"Target Answer: {target_answer}")
                        print(f"Model Response: {output.strip()}")
                        print(f"Match: {success}")
                        
                        # Calculate exact sleep time needed to maintain 14 Requests Per Minute (4.3 seconds per request)
                        elapsed_time = time.time() - request_start_time
                        sleep_time = max(0, 4.3 - elapsed_time)
                        time.sleep(sleep_time)
                        
                        break
                        
                    except Exception as e:
                        error_str = str(e)
                        if "503" in error_str or "429" in error_str:
                            print(f"\n   [!] API Busy. Waiting 15 seconds to retry...", flush=True)
                            time.sleep(15)
                        else:
                            print(f"\nError: {error_str}")
                            break
                
        accuracy = (correct_matches / NUM_QUESTIONS_TO_TEST) * 100
        print(f"\n>> Position {pos} Accuracy: {accuracy:.1f}% ({correct_matches}/{NUM_QUESTIONS_TO_TEST})")
        
        # Save to our results list for Seaborn
        results_data.append({
            "Context Size (Documents)": context_size,
            "Position of Answer (Gold Index)": pos,
            "Accuracy (%)": accuracy
        })

print("\n" + "="*60 + "\nGenerating Seaborn Graph...")

# Convert results to DataFrame
df_results = pd.DataFrame(results_data)

# Draw the graph using Seaborn
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create a line plot with points
sns.lineplot(
    data=df_results, 
    x="Position of Answer (Gold Index)", 
    y="Accuracy (%)", 
    hue="Context Size (Documents)", 
    marker="o",
    palette="tab10",
    linewidth=2.5,
    markersize=8
)

# Formatting the graph
plt.title(f"'Lost in the Middle' Phenomenon\nAccuracy by Answer Position (Gemini API, {NUM_QUESTIONS_TO_TEST} qs/pos)", fontsize=14, pad=15)
plt.ylim(0, 105)
plt.xticks(range(0, 31, 2))
plt.xlabel("Position of Relevant Document (Gold Index)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.legend(title="Context Size")

# Save to file
output_image = "results/gemma_results/gemma_4_31b_it_graph.png"
plt.tight_layout()
plt.savefig(output_image, dpi=300)

print(f"DONE! The graph has been saved to: {output_image}")
print(f"Total API Requests Sent: {total_api_requests}")