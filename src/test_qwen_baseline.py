import os
import sys
import json
import gzip
import string
import re
import time
import ollama

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Force write to disk instantly

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Set up results directory
os.makedirs("results/qwen2.5_7b", exist_ok=True)
sys.stdout = Logger("results/qwen2.5_7b/live_output_baseline.txt")

MODEL_NAME = "qwen2.5:7b-instruct"
NUM_QUESTIONS_TO_TEST = 2655 
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

print(f"Testing CLOSED-BOOK BASELINE (Qwen 2.5 7B) for {NUM_QUESTIONS_TO_TEST} Questions...\n" + "="*60)

# We use one of the dataset files to retrieve the 2,655 test questions
filepath = "data/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz"

correct_matches = 0
all_responses = []
start_time = time.time()

# Check if model is already pulled, otherwise pull it
try:
    print(f"Checking if model '{MODEL_NAME}' is available locally...")
    models_list = ollama.list()
    available_models = [m['name'] for m in models_list.get('models', [])]
    if MODEL_NAME not in available_models and (MODEL_NAME + ":latest") not in available_models:
         print(f"Model '{MODEL_NAME}' not found. Pulling model...")
         ollama.pull(MODEL_NAME)
except Exception as e:
    print(f"Warning checking/pulling model: {e}. Attempting to proceed anyway...")

with gzip.open(filepath, "rt", encoding="utf-8") as f:
    for q_idx in range(NUM_QUESTIONS_TO_TEST):
        line = f.readline()
        if not line:
            break
        data = json.loads(line)
        
        target_query = data["question"]
        target_answers = data["answers"]
        
        # PROMPT WITHOUT ANY DOCUMENTS (Closed-Book Baseline)
        prompt = (
            "Write a high-quality answer for the given question.\n\n"
            f"Question: {target_query}\n"
            "Answer: (Keep the answer as concise as possible, preferably one or two words. Do not write full sentences.)"
        )
        
        try:
            total_api_requests += 1
            q_start = time.time()
            
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_ctx': 4096,
                    'temperature': 0.0
                }
            )
            q_time = time.time() - q_start
            raw_output = response['message']['content']
            
            # Apply first-line truncation (best_subspan_em design)
            output = raw_output.split("\n")[0].strip()
            normalized_prediction = normalize_answer(output)
            
            success = any(
                normalize_answer(ans) in normalized_prediction 
                for ans in target_answers
            )
            
            if success:
                correct_matches += 1
            
            all_responses.append({
                "q_idx": q_idx,
                "question": target_query,
                "target_answers": target_answers,
                "raw_output": raw_output.strip(),
                "truncated_output": output,
                "match": success,
                "time_sec": round(q_time, 2)
            })

            # Print progress log
            if (q_idx + 1) % 50 == 0 or q_idx < 3:
                accuracy_so_far = (correct_matches / (q_idx + 1)) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (q_idx + 1)) * (NUM_QUESTIONS_TO_TEST - q_idx - 1)
                print(f"[Q{q_idx+1}/{NUM_QUESTIONS_TO_TEST}] ({q_time:.1f}s) "
                      f"Acc: {accuracy_so_far:.1f}% | "
                      f"ETA: {eta/60:.0f}min | "
                      f"{target_answers[0]} -> {output[:40]} | {success}")

            # Save intermediate results
            if (q_idx + 1) % 100 == 0 or q_idx == NUM_QUESTIONS_TO_TEST - 1:
                with open("results/qwen2.5_7b/baseline_responses.json", "w") as rf:
                    json.dump(all_responses, rf, indent=2)

        except Exception as e:
            print(f"\nError running generation or connecting to Ollama: {str(e)}")
            print("Make sure you have Ollama installed and running in the background!")
            break

accuracy = (correct_matches / total_api_requests) * 100 if total_api_requests > 0 else 0.0
print(f"\n{'='*40}")
print(f">> BASELINE ACCURACY: {accuracy:.2f}% ({correct_matches}/{total_api_requests})")
print(f"Total API Requests Sent: {total_api_requests}")
print(f"{'='*40}")
