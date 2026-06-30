import json
from pathlib import Path

eval_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')
answers_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')

DEGENERATE_QIDS = {
    '2001908', '2003157', '2003976', '2026150', '2027130', '2027497',
    '2033470', '2034676', '2040352', '2044323', '2046027', '2051782',
    '3010623', '3100188', '3100292', '421946', '818583'
}

# Check condition 4 for suspicious 0-recall queries on non-degenerate answers
for c in [4, 5]:
    fp_eval = eval_dir / f'k60_{c}_eval.json'
    fp_ans  = answers_dir / f'k60_{c}.jsonl'
    if not fp_eval.exists(): continue

    eval_data = json.loads(fp_eval.read_text(encoding='utf-8'))
    rows = {str(r['query_id']): r for r in
            [json.loads(l) for l in fp_ans.read_text(encoding='utf-8').splitlines() if l.strip()]}

    suspicious = []
    for qid, er in eval_data.items():
        if qid in DEGENERATE_QIDS: continue
        if er['vital_recall'] == 0 and er['total_vital'] > 0:
            ans = rows.get(qid, {}).get('qwen_answer', '').strip()
            # Good answer but 0 vital recall = possibly failed API call
            if ans and len(set(ans[:100])) > 5:
                suspicious.append((qid, er['total_vital'], ans[:80]))

    print(f"\nCondition {c}: {len(suspicious)} queries with 0% vital recall but GOOD answers (possible API failures):")
    for qid, nvital, ans_preview in suspicious[:10]:
        print(f"  Query {qid} ({nvital} vital nuggets): {ans_preview}")
