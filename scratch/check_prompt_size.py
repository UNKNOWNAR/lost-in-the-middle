import json
from pathlib import Path

answers_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')
nuggets_path = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\nuggets.json')
all_nuggets = json.loads(nuggets_path.read_text(encoding='utf-8'))

# Check the first query in condition 1 (23287 - the one that's failing)
fp = answers_dir / 'k60_1.jsonl'
rows = [json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines() if l.strip()]

DEGENERATE_QIDS = {
    '2001908', '2003157', '2003976', '2026150', '2027130', '2027497',
    '2033470', '2034676', '2040352', '2044323', '2046027', '2051782',
    '3010623', '3100188', '3100292', '421946', '818583'
}
valid_rows = [r for r in rows if str(r['query_id']) not in DEGENERATE_QIDS]

print(f"Total valid queries in condition 1: {len(valid_rows)}")
print(f"\nFirst 5 query IDs: {[str(r['query_id']) for r in valid_rows[:5]]}")
print()

for r in valid_rows[:3]:
    qid = str(r['query_id'])
    ans = r.get('qwen_answer', '').strip()
    nuggets_list = all_nuggets.get(qid, {}).get('all_nuggets', [])
    nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])
    full_prompt = f"MODEL ANSWER:\n{ans}\n\nFACTUAL NUGGETS:\n{nuggets_text}"
    # Rough token estimate: 1 token ~ 4 chars
    approx_tokens = len(full_prompt) // 4
    print(f"Query {qid}:")
    print(f"  Answer length: {len(ans)} chars")
    print(f"  Nuggets: {len(nuggets_list)}")
    print(f"  Total prompt chars: {len(full_prompt)}")
    print(f"  Approx tokens: ~{approx_tokens}")
    print()
