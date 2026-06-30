import json
for i in range(1, 5):
    try:
        path = f'e:/WorkSpace/lostinthemiddle/litm_pipeline/data/processed/evaluations/k60_{i}_eval.json'
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if "2056158" in data:
                print(f'Condition {i}: query 2056158 vital_recall = {data["2056158"]["vital_recall"]}%')
            else:
                print(f'Condition {i}: query 2056158 not in results')
    except Exception as e:
        print(f'Condition {i}: Error {e}')
