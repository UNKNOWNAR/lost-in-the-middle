import json
from pathlib import Path

eval_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')

for c in range(1, 11):
    fp = eval_dir / f'k60_{c}_eval.json'
    if not fp.exists():
        continue
    try:
        data = json.loads(fp.read_text(encoding='utf-8'))
        if len(data) < 119:
            continue  # skip partial
        vital_recalls = [d.get('vital_recall', 0) for d in data.values()]
        okay_recalls = [d.get('okay_recall', 0) for d in data.values()]
        avg_vital = sum(vital_recalls) / len(vital_recalls)
        avg_okay = sum(okay_recalls) / len(okay_recalls)
        print(f'Condition {c}: Vital Recall = {avg_vital:.2f}%  |  Okay Recall = {avg_okay:.2f}%')
    except Exception as e:
        print(f'Condition {c}: Error - {e}')
