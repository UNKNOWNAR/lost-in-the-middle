import os
import json
import statistics
from collections import defaultdict

RAW_DIR = r"E:\WorkSpace\lostinthemiddle\litm_pipeline\data\raw"
PROCESSED_DIR = r"E:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed"

def main():
    bm25_path = os.path.join(RAW_DIR, "bm25.top1000.raggy-dev.jsonl")
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 2. Parse BM25 JSONL
    print("Loading BM25 candidates...")
    queries = {}
    per_query_passages = {}
    qrels = defaultdict(dict)
    
    with open(bm25_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data["qid"])
            qtext = data["query"]
            
            queries[qid] = qtext
            
            # The file contains qrels directly!
            if "qrels" in data:
                for docid, rel in data["qrels"].items():
                    qrels[qid][docid] = int(rel)
            
            golden_passages = []
            distractor_passages = []
            
            for candidate in data.get("top1000", []):
                docid = candidate.get("docid")
                score = candidate.get("bm25_score")
                rank = candidate.get("rank")
                text = candidate.get("text", "")
                
                base_docid = docid.split("#")[0] if docid else ""
                rel = qrels.get(qid, {}).get(base_docid, 0)
                
                passage_info = {
                    "docid": docid,
                    "qrel": rel,
                    "bm25_score": score,
                    "bm25_rank": rank,
                    "text": text
                }
                
                if rel >= 2:
                    golden_passages.append(passage_info)
                else:
                    distractor_passages.append(passage_info)
            
            per_query_passages[qid] = {
                "query_text": qtext,
                "golden_passages": golden_passages,
                "distractor_passages": distractor_passages,
                "n_golden": len(golden_passages),
                "n_distractors": len(distractor_passages)
            }

    # 3. Calculate and print statistics
    golden_counts = [info["n_golden"] for info in per_query_passages.values()]
    distractor_counts = [info["n_distractors"] for info in per_query_passages.values()]
    
    print("\n--- Statistics for Golden Docs (Qrels >= 2) per query ---")
    print(f"Min: {min(golden_counts)}")
    print(f"Max: {max(golden_counts)}")
    print(f"Avg: {statistics.mean(golden_counts):.2f}")
    print(f"Median: {statistics.median(golden_counts)}")
    try:
        print(f"Mode: {statistics.mode(golden_counts)}")
    except statistics.StatisticsError:
        print("Mode: Multiple modes found")
        
    print("\n--- Statistics for Distractor Docs per query ---")
    print(f"Min: {min(distractor_counts)}")
    print(f"Max: {max(distractor_counts)}")
    print(f"Avg: {statistics.mean(distractor_counts):.2f}")
    print(f"Median: {statistics.median(distractor_counts)}")
    try:
        print(f"Mode: {statistics.mode(distractor_counts)}")
    except statistics.StatisticsError:
        print("Mode: Multiple modes found")

    # 4. Save to processed
    queries_out = os.path.join(PROCESSED_DIR, "queries.json")
    with open(queries_out, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2)
        
    passages_out = os.path.join(PROCESSED_DIR, "per_query_passages.json")
    with open(passages_out, 'w', encoding='utf-8') as f:
        json.dump(per_query_passages, f, indent=2)
        
    qrels_map_out = os.path.join(PROCESSED_DIR, "qrels_map.json")
    with open(qrels_map_out, 'w', encoding='utf-8') as f:
        json.dump(qrels, f, indent=2)

    print(f"\nSaved processed data to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
