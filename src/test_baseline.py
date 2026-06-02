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
        self.log.flush() # Force write to disk instantly

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("live_output_baseline.txt")

MODEL_NAME = "llama3.1"

# Testing all 2655 questions!
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

print(f"Testing CLOSED-BOOK BASELINE for {NUM_QUESTIONS_TO_TEST} Questions...\n" + "="*60)

# We just need to read one file to get all 2655 questions. 
# We will completely ignore the documents inside it.
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
        
        # PROMPT WITHOUT ANY DOCUMENTS (Closed-Book Baseline)
        prompt = (
            "Write a high-quality answer for the given question.\n\n"
            f"Question: {target_query}\n"
            "Answer: (Keep the answer as concise as possible, preferably one or two words. Do not write full sentences.)"
        )
        
        try:
            total_api_requests += 1
            request_start_time = time.time()
            
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_ctx': 4096,
                    'temperature': 0.0,
                    'num_gpu': 20
                }
            )
            output = response['message']['content']
            
            normalized_prediction = normalize_answer(output)
            normalized_ground_truth = normalize_answer(target_answer)
            
            success = normalized_ground_truth in normalized_prediction
            if success:
                correct_matches += 1
            
            print(f"\n[Question {q_idx+1}]")
            print(f"Target Answer: {target_answer}")
            print(f"Model Response: {output.strip()}")
            print(f"Match: {success}")
            
        except Exception as e:
            print(f"\nError connecting to Ollama: {str(e)}")
            print("Make sure you have Ollama installed and running in the background!")
            break

accuracy = (correct_matches / NUM_QUESTIONS_TO_TEST) * 100
print(f"\n{'='*40}")
print(f">> BASELINE ACCURACY: {accuracy:.1f}% ({correct_matches}/{NUM_QUESTIONS_TO_TEST})")
print(f"Total API Requests Sent: {total_api_requests}")
print(f"{'='*40}")
