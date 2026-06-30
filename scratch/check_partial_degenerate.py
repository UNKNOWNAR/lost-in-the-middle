import json
from pathlib import Path

answers_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')

for qid, label in [('421946', '9/10'), ('818583', '8/10')]:
    print(f"\n{'='*60}")
    print(f"Query {qid} (degenerate in {label} conditions)")
    print(f"{'='*60}")
    for c in range(1, 11):
        fp = answers_dir / f'k60_{c}.jsonl'
        if not fp.exists():
            print(f"  Condition {c}: FILE NOT FOUND")
            continue
        rows = [json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines() if l.strip()]
        row = next((r for r in rows if str(r['query_id']) == qid), None)
        if not row:
            print(f"  Condition {c}: QUERY NOT IN FILE")
            continue
        ans = row.get('qwen_answer', '').strip()
        is_deg = not ans or len(set(ans[:50])) <= 3
        status = "DEGENERATE" if is_deg else "NORMAL"
        print(f"  Condition {c}: [{status}] Answer[:80]: {repr(ans[:80])}")
