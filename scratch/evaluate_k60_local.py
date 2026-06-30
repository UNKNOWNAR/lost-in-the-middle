import os
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(r'e:\WorkSpace\lostinthemiddle\.env'))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

NUGGETS_PATH = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\nuggets.json')
INPUT_DIR = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')
OUTPUT_DIR = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_nuggets = json.loads(NUGGETS_PATH.read_text(encoding='utf-8'))

PROMPT_TEMPLATE = """You are a strict academic evaluator. 
Given a MODEL ANSWER and a list of FACTUAL NUGGETS, determine whether the MODEL ANSWER contains the information in each nugget.

MODEL ANSWER:
{model_answer}

FACTUAL NUGGETS:
{nuggets_text}

Output compact single-line JSON only (no markdown, no newlines inside JSON, no explanation):
{{"evaluations":[{{"id":1,"covered":true}},{{"id":2,"covered":false}}]}}
"""

def evaluate_query(query_id, model_answer, nuggets_list):
    nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])
    prompt = PROMPT_TEMPLATE.format(model_answer=model_answer, nuggets_text=nuggets_text)
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8000,
            )
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data = json.loads(raw)
        evaluations = data.get("evaluations", [])
        return {item["id"]: item.get("covered", False) for item in evaluations}
    except Exception as e:
        print(f"Error evaluating query {query_id}: {e}")
        return {}

def main():
    files = sorted(INPUT_DIR.glob("k60_*.jsonl"), key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
    
    for fp in files:
        c_idx = fp.stem.split("_")[-1]
        out_f = OUTPUT_DIR / f"k60_{c_idx}_eval.json"
        
        if out_f.exists():
            print(f"Skipping {fp.name}, output exists.")
            continue
            
        print(f"Evaluating {fp.name}...")
        results = {}
        rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        
        for i, d in enumerate(rows):
            qid = str(d["query_id"])
            qtxt = d.get("query_text", "")
            ans = d.get("qwen_answer", "").strip()
            
            if qid not in all_nuggets: continue
            nuggets_list = all_nuggets[qid]["all_nuggets"]
            if not nuggets_list: continue
            
            # Fast-path for degenerate answers
            if not ans or len(set(ans[:50])) <= 3:
                eval_map = {n["id"]: False for n in nuggets_list}
            else:
                eval_map = evaluate_query(qid, ans, nuggets_list)
                # Avoid rate limit
                time.sleep(1)
                
            query_vital = 0
            query_covered_vital = 0
            query_okay = 0
            query_covered_okay = 0
            nugget_results = []
            
            for n in nuggets_list:
                nid = n["id"]
                is_covered = eval_map.get(nid, False)
                is_vital = (n["vitality"] == "vital")
                
                if is_vital:
                    query_vital += 1
                    if is_covered:
                        query_covered_vital += 1
                else:
                    query_okay += 1
                    if is_covered:
                        query_covered_okay += 1
                        
                nugget_results.append({
                    "id": nid,
                    "text": n["text"],
                    "vitality": n["vitality"],
                    "covered": is_covered
                })

            results[qid] = {
                "query_id": qid,
                "query_text": qtxt,
                "total_vital": query_vital,
                "covered_vital": query_covered_vital,
                "vital_recall": (query_covered_vital / query_vital * 100) if query_vital > 0 else 0,
                "total_okay": query_okay,
                "covered_okay": query_covered_okay,
                "okay_recall": (query_covered_okay / query_okay * 100) if query_okay > 0 else 0,
                "nugget_results": nugget_results
            }
            if i > 0 and i % 10 == 0:
                print(f"  Processed {i}/{len(rows)}...")
                
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Finished {fp.name}")

if __name__ == "__main__":
    main()
