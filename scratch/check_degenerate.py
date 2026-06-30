import json
from pathlib import Path

answers_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')

# Find degenerate queries in each condition
degenerate_by_condition = {}
all_qids = set()

for c in range(1, 11):
    fp = answers_dir / f'k60_{c}.jsonl'
    if not fp.exists():
        print(f"Condition {c}: file not found")
        continue
    rows = [json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines() if l.strip()]
    degenerate = set()
    for r in rows:
        ans = r.get('qwen_answer', '').strip()
        qid = str(r['query_id'])
        all_qids.add(qid)
        if not ans or len(set(ans[:50])) <= 3:
            degenerate.add(qid)
    degenerate_by_condition[c] = degenerate
    print(f"Condition {c}: {len(degenerate)} degenerate queries")

# Find queries degenerate in ALL available conditions
available = sorted(degenerate_by_condition.keys())
always_degenerate = degenerate_by_condition[available[0]].copy()
for c in available[1:]:
    always_degenerate &= degenerate_by_condition[c]

sometimes_degenerate = set()
for c in available:
    sometimes_degenerate |= degenerate_by_condition[c]

print(f"\n--- Summary ---")
print(f"Queries degenerate in ALL {len(available)} available conditions: {len(always_degenerate)}")
print(f"Queries degenerate in AT LEAST ONE condition: {len(sometimes_degenerate)}")
print(f"\nQuery IDs always degenerate: {sorted(always_degenerate)}")

# Check per-query how many conditions they are degenerate in
print("\n--- Per-query degenerate count ---")
for qid in sorted(sometimes_degenerate):
    count = sum(1 for c in available if qid in degenerate_by_condition[c])
    print(f"  Query {qid}: degenerate in {count}/{len(available)} conditions")
