import json
import os

removals = {
    1: ["2001010"],
    2: ["2007055", "2007419", "2014738", "2030323"],
    3: ["2001459", "2007419", "2014738", "2056158"],
    4: ["2007419", "2014738", "2030323", "2056158"]
}

for i in range(1, 5):
    path = f"e:/WorkSpace/lostinthemiddle/litm_pipeline/data/processed/evaluations/k60_{i}_eval.json"
    if not os.path.exists(path):
        continue
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    cleaned = False
    for qid in removals.get(i, []):
        if qid in data:
            print(f"Condition {i}: Accurately removing false zero for safety-blocked query {qid}")
            del data[qid]
            cleaned = True
                
    if cleaned:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
