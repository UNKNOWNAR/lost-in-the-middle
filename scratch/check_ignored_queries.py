import json
from pathlib import Path
from collections import defaultdict

def analyze_ignored_queries():
    # Load all 102 queries to get their text
    with open('litm_pipeline/data/processed/queries.json', 'r', encoding='utf-8') as f:
        queries_dict = json.load(f)
    
    # Track missing counts
    missing_counts = defaultdict(int)
    
    # Path to the condition outputs
    results_dir = Path('litm_pipeline/data/processed/evaluations')
    
    # Read each condition file
    for i in range(1, 11):
        file_path = results_dir / f"eval_k60_{i}_gemma.json"
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            evaluated_data = json.load(f)
            # evaluated_data is a list of dicts, each with 'query_id'
            evaluated_qids = {str(item['query_id']) for item in evaluated_data}
            
            # Check which of the 102 valid queries are missing
            for qid in queries_dict.keys():
                if qid not in evaluated_qids:
                    missing_counts[qid] += 1
                    
    # Sort by frequency (highest missing count first)
    sorted_missing = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("=== Consistently Ignored / Skipped Queries ===")
    print(f"{'Query ID':<10} | {'Times Skipped':<15} | {'Query Text'}")
    print("-" * 80)
    
    for qid, count in sorted_missing:
        # Only show queries that were skipped at least once
        text = queries_dict.get(qid, "Unknown text")
        print(f"{qid:<10} | {count:<2} out of 10   | {text}")

if __name__ == "__main__":
    analyze_ignored_queries()
