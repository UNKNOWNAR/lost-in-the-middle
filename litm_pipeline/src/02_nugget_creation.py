import os
import json
from pathlib import Path
from utils import (
    GeminiClient,
    setup_logging,
    load_prompt,
    DATA_PROCESSED,
    save_checkpoint,
    load_checkpoint
)

logger = setup_logging("02_nugget_creation")

def normalize_text(text):
    return " ".join(text.lower().split())

def process_query(query_id, data, client, prompt_template):
    query_text = data["query_text"]
    golden_passages = data["golden_passages"]
    
    # We select the top 3 gold (relevance >= 2) passages to act as references.
    # The list should already be sorted by BM25, but let's take the first 3.
    top_golden = golden_passages[:3]
    
    passages_text = []
    for i, p in enumerate(top_golden, start=1):
        passages_text.append(f"[Golden Doc {i}] Title: {p['title']}\n{p['text']}")
    
    golden_passages_text = "\n---\n".join(passages_text)
    
    prompt = prompt_template.format(
        query_text=query_text,
        golden_passages_text=golden_passages_text
    )
    
    try:
        nuggets = client.generate_json(prompt, max_tokens=800)
    except Exception as e:
        logger.error(f"Failed to extract nuggets for {query_id}: {e}")
        return None
    
    if not isinstance(nuggets, list):
        logger.error(f"Response for {query_id} is not a list. Skipping.")
        return None
        
    # Deduplicate nuggets by text
    seen = set()
    unique_nuggets = []
    for n in nuggets:
        if not isinstance(n, dict): continue
        n_text = normalize_text(n.get("text", ""))
        if n_text not in seen:
            seen.add(n_text)
            unique_nuggets.append(n)
            
    # Assign standard incremental IDs
    for i, n in enumerate(unique_nuggets, start=1):
        n["id"] = i

    vital_nuggets = [n for n in unique_nuggets if n.get("vitality") == "vital"]
    okay_nuggets = [n for n in unique_nuggets if n.get("vitality") == "okay"]
    
    result = {
        "query_text": query_text,
        "all_nuggets": unique_nuggets,
        "vital_nuggets": vital_nuggets,
        "okay_nuggets": okay_nuggets,
        "total_count": len(unique_nuggets),
        "vital_count": len(vital_nuggets),
        "n_golden_docs_used": len(top_golden)
    }
    
    return result

def main():
    passages_path = DATA_PROCESSED / "per_query_passages.json"
    nuggets_path = DATA_PROCESSED / "nuggets.json"
    
    with open(passages_path, "r", encoding="utf-8") as f:
        per_query_passages = json.load(f)
        
    prompt_template = load_prompt("nugget_creation_prompt.txt")
    client = GeminiClient()
    
    # Load existing nuggets to resume if interrupted
    all_nuggets = load_checkpoint(nuggets_path) or {}
    
    logger.info(f"Loaded {len(per_query_passages)} queries.")
    
    count = 0
    for query_id, data in per_query_passages.items():
        if query_id in all_nuggets:
            continue
            
        logger.info(f"Processing query {query_id}...")
        result = process_query(query_id, data, client, prompt_template)
        
        if result:
            all_nuggets[query_id] = result
            logger.info(f"Query {query_id}: {result['total_count']} nuggets ({result['vital_count']} vital)")
        else:
            logger.warning(f"Skipping query {query_id} due to failure.")
            
        count += 1
        # Save checkpoint every 10 queries
        if count % 10 == 0:
            logger.info("Saving checkpoint...")
            save_checkpoint(all_nuggets, nuggets_path)
            
    # Final save
    save_checkpoint(all_nuggets, nuggets_path)
    logger.info("Done extracting nuggets.")

if __name__ == "__main__":
    main()
