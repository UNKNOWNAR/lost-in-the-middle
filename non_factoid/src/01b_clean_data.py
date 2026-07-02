import json
import os

PROCESSED_DIR = r"E:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed"

def main():
    """
    Utility script to clean the processed data.
    Removes any queries that lack golden documents (n_golden == 0).
    This ensures that Phase 2 (Nugget Extraction) has the required reference material.
    """
    passages_path = os.path.join(PROCESSED_DIR, "per_query_passages.json")
    queries_path = os.path.join(PROCESSED_DIR, "queries.json")
    
    if not os.path.exists(passages_path):
        print("Processed data not found. Please run 01_data_preparation.py first.")
        return

    with open(passages_path, "r", encoding="utf-8") as f:
        per_query_passages = json.load(f)
        
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Identify queries with 0 golden documents
    keys_to_delete = [qid for qid, data in per_query_passages.items() if data['n_golden'] == 0]
    
    if not keys_to_delete:
        print("No queries with 0 golden documents found. Data is clean.")
        return
        
    print(f"Found {len(keys_to_delete)} queries with 0 golden docs: {keys_to_delete}")
    
    # Remove them
    for qid in keys_to_delete:
        per_query_passages.pop(qid, None)
        queries.pop(qid, None)
        
    # Save the cleaned data back
    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(per_query_passages, f, indent=2)
        
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
        
    print("Cleaned data successfully saved.")

if __name__ == "__main__":
    main()
