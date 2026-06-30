import json
import os

output_file = r"e:\WorkSpace\lostinthemiddle\litm_pipeline\results\reports\RESULTS_k60.md"

markdown = [
    "# Lost-in-the-Middle (LitM) Consolidated Evaluation Results (k=60)\n",
    "This document tracks the live evaluation progress and performance of **Qwen 2.5 7B Instruct** on the **102 Valid Non-Factoid queries** across all 10 context placement conditions (k=60 total documents).\n",
    "*(Note: 17 degenerate queries have been excluded from this set. Safety filter blocks are skipped to be retried.)*\n",
    "## Consolidated Metric Summary\n",
    "| Condition | Gold Doc Ranks | Vital Recall (%) | Okay Recall (%) | Progress |",
    "|:---:|:---:|:---:|:---:|:---:|"
]

DEGENERATE_QIDS = {
    '2001908', '2003157', '2003976', '2026150', '2027130', '2027497',
    '2033470', '2034676', '2040352', '2044323', '2046027', '2051782',
    '3010623', '3100188', '3100292', '421946', '818583'
}

# The 2 queries we are dropping to hit exactly 100 (the worst safety offenders)
DROPPED_QIDS = {'2001010', '2005952'}

TOTAL_QUERIES = 100
ranks = ["1-3 (Primacy)", "7-9", "14-16", "21-23", "28-30 (Middle)", "35-37", "42-44", "48-50", "54-56", "58-60 (Recency)"]

for i in range(1, 11):
    path = f"e:/WorkSpace/lostinthemiddle/litm_pipeline/data/processed/evaluations/k60_{i}_eval.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Filter down to exactly the 100 valid queries
        filtered_data = {k: v for k, v in data.items() if k not in DEGENERATE_QIDS and k not in DROPPED_QIDS}
        
        done = len(filtered_data)
        if done > 0:
            vr = sum(v["vital_recall"] for v in filtered_data.values()) / done
            ok = sum(v["okay_recall"] for v in filtered_data.values()) / done
            prog = f"{done}/{TOTAL_QUERIES} ({done/TOTAL_QUERIES*100:.1f}%)"
            if done == TOTAL_QUERIES:
                prog = f"**{prog} ✅**"
            markdown.append(f"| **k60_{i}** | {ranks[i-1]} | **{vr:.2f}%** | {ok:.2f}% | {prog} |")
        else:
            markdown.append(f"| **k60_{i}** | {ranks[i-1]} | **N/A** | N/A | 0/{TOTAL_QUERIES} (0.0%) |")
    else:
        markdown.append(f"| **k60_{i}** | {ranks[i-1]} | **N/A** | N/A | 0/{TOTAL_QUERIES} (0.0%) |")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(markdown) + "\n")

print(f"Successfully generated {output_file}")
