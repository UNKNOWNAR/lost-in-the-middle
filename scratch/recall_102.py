import json
from pathlib import Path

eval_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')

DEGENERATE_QIDS = {
    '2001908', '2003157', '2003976', '2026150', '2027130', '2027497',
    '2033470', '2034676', '2040352', '2044323', '2046027', '2051782',
    '3010623', '3100188', '3100292', '421946', '818583'
}

print("Recall results — 102 valid queries (17 degenerate excluded)")
print("=" * 60)
print(f"{'Condition':<12} {'N':<6} {'Vital Recall':>14} {'Okay Recall':>12}")
print("-" * 60)

for c in range(1, 11):
    fp = eval_dir / f'k60_{c}_eval.json'
    if not fp.exists():
        continue
    try:
        data = json.loads(fp.read_text(encoding='utf-8'))
    except:
        continue

    filtered = {qid: d for qid, d in data.items() if qid not in DEGENERATE_QIDS}
    n = len(filtered)
    if n == 0:
        continue

    vital_recalls = [d['vital_recall'] for d in filtered.values()]
    okay_recalls  = [d['okay_recall']  for d in filtered.values()]
    avg_vital = sum(vital_recalls) / n
    avg_okay  = sum(okay_recalls)  / n

    complete = "DONE" if len(data) == 119 else f"partial {len(data)}/119"
    print(f"Condition {c:<3} {n:<6} {avg_vital:>13.2f}% {avg_okay:>11.2f}%   ({complete})")

print("-" * 60)
print("(Only fully complete conditions shown above)")
