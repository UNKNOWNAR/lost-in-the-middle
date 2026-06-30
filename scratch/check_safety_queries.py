import re

log_files = [
    r'C:\Users\amiar\.gemini\antigravity-ide\brain\187fbd4c-3682-48f3-9b86-a66cec767b8e\.system_generated\tasks\task-6408.log',
    r'C:\Users\amiar\.gemini\antigravity-ide\brain\187fbd4c-3682-48f3-9b86-a66cec767b8e\.system_generated\tasks\task-6475.log'
]

safety_flagged = set()

for log in log_files:
    try:
        with open(log, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Empty/safety response' in line:
                    match = re.search(r'\[(\d+)\] Failed', line)
                    if match:
                        safety_flagged.add(match.group(1))
    except FileNotFoundError:
        pass

print("=== Queries flagged by Google Safety Filters ===")
import json
with open('litm_pipeline/data/processed/queries.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

for qid in sorted(list(safety_flagged)):
    text = queries.get(qid, "Unknown text")
    print(f"Query {qid}: {text}")
