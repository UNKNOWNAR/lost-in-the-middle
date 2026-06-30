import json
from pathlib import Path

eval_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')
answers_dir = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')
nuggets_path = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\nuggets.json')
all_nuggets = json.loads(nuggets_path.read_text(encoding='utf-8'))

print("=" * 60)
print("DIAGNOSTIC: k=60 Condition 1 deep inspection")
print("=" * 60)

# 1. Check how many answers are degenerate in condition 1
fp = answers_dir / 'k60_1.jsonl'
rows = [json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines() if l.strip()]
degenerate = [r for r in rows if len(set(r.get('qwen_answer','')[:50])) <= 3 or not r.get('qwen_answer','').strip()]
print(f"\n1. Degenerate answers in k60_1.jsonl: {len(degenerate)}/{len(rows)}")

# 2. Show 3 sample answers (good ones)
good = [r for r in rows if r not in degenerate]
print(f"\n2. Sample of a GOOD answer (first 300 chars):")
if good:
    sample = good[0]
    print(f"   Query: {sample.get('query_text','')[:80]}")
    print(f"   Answer: {sample.get('qwen_answer','')[:300]}")

# 3. Check the eval result for that same query
eval_fp = eval_dir / 'k60_1_eval.json'
eval_data = json.loads(eval_fp.read_text(encoding='utf-8'))
qid = str(sample['query_id'])
if qid in eval_data:
    er = eval_data[qid]
    print(f"\n3. Eval result for query {qid}:")
    print(f"   Vital: {er['covered_vital']}/{er['total_vital']} = {er['vital_recall']:.1f}%")
    print(f"   Okay:  {er['covered_okay']}/{er['total_okay']} = {er['okay_recall']:.1f}%")
    print(f"   Nuggets:")
    for n in er['nugget_results'][:5]:
        print(f"     [{n['vitality']}] covered={n['covered']}: {n['text'][:80]}")

# 4. Compare with k=20 and k=40 condition 1 recalls
print("\n4. Comparing Condition 1 across k values:")
for k in [20, 40, 60]:
    kfp = eval_dir / f'k{k}_1_eval.json'
    if kfp.exists():
        kdata = json.loads(kfp.read_text(encoding='utf-8'))
        vr = [d['vital_recall'] for d in kdata.values()]
        ok = [d['okay_recall'] for d in kdata.values()]
        print(f"   k={k}: Vital={sum(vr)/len(vr):.2f}%  Okay={sum(ok)/len(ok):.2f}%  (n={len(vr)})")

# 5. Check what percent of nuggets are covered per query in k60_1
print("\n5. Distribution of vital recall in k60_1 (how many queries got 0%, >50%, 100%):")
vr_vals = [eval_data[q]['vital_recall'] for q in eval_data]
zero = sum(1 for v in vr_vals if v == 0)
low = sum(1 for v in vr_vals if 0 < v <= 50)
high = sum(1 for v in vr_vals if v > 50)
print(f"   0% recall: {zero} queries")
print(f"   1-50% recall: {low} queries")
print(f"   >50% recall: {high} queries")

# 6. Look at a query that got 0 vital recall
zero_queries = [q for q in eval_data if eval_data[q]['vital_recall'] == 0 and eval_data[q]['total_vital'] > 0]
if zero_queries:
    qid0 = zero_queries[0]
    er0 = eval_data[qid0]
    # find the answer
    ans_row = next((r for r in rows if str(r['query_id']) == qid0), None)
    print(f"\n6. A query with 0% vital recall: {qid0}")
    if ans_row:
        print(f"   Query: {ans_row.get('query_text','')[:80]}")
        print(f"   Answer (first 300): {ans_row.get('qwen_answer','')[:300]}")
    print(f"   Vital nuggets:")
    for n in er0['nugget_results']:
        if n['vitality'] == 'vital':
            print(f"     covered={n['covered']}: {n['text'][:80]}")
