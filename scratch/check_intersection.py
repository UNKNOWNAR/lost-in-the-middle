import json
from pathlib import Path
from collections import defaultdict

def find_intersection():
    results_dir = Path('litm_pipeline/data/processed/evaluations')
    
    # Track success count for each query
    success_counts = defaultdict(int)
    
    for i in range(1, 11):
        file_path = results_dir / f"k60_{i}_eval.json"
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            evaluated_data = json.load(f)
            # data is a dict of qid -> result
            for qid in evaluated_data.keys():
                success_counts[str(qid)] += 1
                
    # Find queries that succeeded in all 10 conditions
    perfect_queries = [qid for qid, count in success_counts.items() if count == 10]
    
    print(f"Total unique queries evaluated at least once: {len(success_counts)}")
    print(f"Queries successful in ALL 10 conditions: {len(perfect_queries)}")
    
    # Print the counts
    count_distribution = defaultdict(int)
    for qid, count in success_counts.items():
        count_distribution[count] += 1
        
    print("\nSuccess Frequency Distribution:")
    for count in sorted(count_distribution.keys(), reverse=True):
        print(f"Succeeded in {count} conditions: {count_distribution[count]} queries")

if __name__ == "__main__":
    find_intersection()
