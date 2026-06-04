import os
import sys
import time
import json
import gzip
import re
import string
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
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

os.makedirs("results/gemma_results", exist_ok=True)
sys.stdout = Logger("results/gemma_results/live_output_gemma_baseline.txt")

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = "gemma-4-31b-it"

NUM_QUESTIONS_TO_TEST = 300
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

print(f"Testing CLOSED-BOOK BASELINE for {NUM_QUESTIONS_TO_TEST} Questions using {MODEL_NAME}...")
print("=" * 60)

# We read one of the datasets to extract the first 300 questions
filepath = "lost-in-the-middle/qa_data/10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz"
correct_matches = 0

with gzip.open(filepath, "rt", encoding="utf-8") as f:
    for q_idx in range(NUM_QUESTIONS_TO_TEST):
        line = f.readline()
        if not line:
            break
        data = json.loads(line)
        
        target_query = data["question"]
        target_answer = data["answers"][0]
        
        # Closed-Book prompt structure without context search results
        prompt = f"""Write a high-quality answer for the given question.

Question: {target_query}
Answer: (Keep the answer as concise as possible, preferably one or two words. Do not write full sentences.)"""

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
                
                print(f"\n[Question {q_idx+1}]")
                print(f"Target Answer: {target_answer}")
                print(f"Model Response: {output.strip()}")
                print(f"Match: {success}")
                
                # Sleep briefly to manage rate limit
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
print(f"\n{'='*40}")
print(f">> BASELINE ACCURACY: {accuracy:.1f}% ({correct_matches}/{NUM_QUESTIONS_TO_TEST})")
print(f"Total API Requests Sent: {total_api_requests}")
print(f"{'='*40}")
