import json
import os

qids = ["2001010", "2001459", "2007055", "2007419", "2014738", "2030323", "2056158"]
results = {qid: {} for qid in qids}

for i in range(1, 5):
    path = f"e:/WorkSpace/lostinthemiddle/litm_pipeline/data/processed/evaluations/k60_{i}_eval.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for qid in qids:
                if qid in data:
                    results[qid][f"Cond_{i}"] = data[qid].get("vital_recall", "N/A")
                else:
                    results[qid][f"Cond_{i}"] = "N/A"

for qid in qids:
    res_str = ", ".join(f"{k}: {v}" for k, v in results[qid].items())
    print(f"Query {qid} -> {res_str}")
