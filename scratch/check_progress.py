import json
from pathlib import Path

eval_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')

total_queries = 0
fully_done = 0
for c in range(1, 11):
    fp = eval_dir / f'k60_{c}_eval.json'
    if not fp.exists():
        print(f'Condition {c}: NOT FOUND')
        continue
    try:
        data = json.loads(fp.read_text(encoding='utf-8'))
        n = len(data)
        total_queries += n
        done = "COMPLETE" if n == 119 else f"PARTIAL ({n}/119)"
        if n == 119:
            fully_done += 1
        print(f'Condition {c}: {done}  ({fp.stat().st_size//1024} KB)')
    except Exception as e:
        print(f'Condition {c}: CORRUPT/WRITING - {e}')

print(f'\nTotal evaluated queries: {total_queries}')
print(f'Fully complete conditions: {fully_done}/10')
print(f'Remaining queries: {10*119 - total_queries}')
