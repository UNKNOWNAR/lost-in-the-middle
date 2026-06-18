"""
06_evaluate_answers.py

Evaluates generated answers (JSONL format) against the ground-truth nuggets.
Uses an LLM as a Judge (Groq/Llama-3 by default) to score whether each nugget is present.
"""

import os
import sys
import json
import time
import re
import logging
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv
import argparse
from google import genai
from google.genai import types
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("06_evaluate_answers")

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"
EVALS_DIR = DATA_PROCESSED / "evaluations"

# Load environment variables
load_dotenv(PIPELINE_ROOT.parent / ".env")
load_dotenv(PIPELINE_ROOT / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=2, initial_delay=2.0)
            )
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")

# Rate Limiter setup (14 RPM margin)
MIN_INTERVAL = 60.0 / 14
last_call_time = 0.0

class NuggetEvaluation(BaseModel):
    id: int
    covered: bool

class AnswerEvaluation(BaseModel):
    evaluations: list[NuggetEvaluation]

def call_gemini_with_rate_limit(prompt: str, model_name: str = "gemini-3.1-flash-lite") -> str:
    global last_call_time
    if not client:
        raise ValueError("Gemini client is not initialized. Check GEMINI_API_KEY.")
    elapsed = time.time() - last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    last_call_time = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1500,
            response_mime_type="application/json",
            response_schema=AnswerEvaluation,
        ),
    )
    return response.text.strip()

last_groq_call_time = 0.0

def call_groq_with_rate_limit(prompt: str) -> str:
    global last_groq_call_time
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please set it in .env")
    
    elapsed = time.time() - last_groq_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    last_groq_call_time = time.time()

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            resp_data = response.read().decode("utf-8")
            resp_json = json.loads(resp_data)
            return resp_json["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"Groq API Error {e.code}: {error_body}")

PROMPT_TEMPLATE = """You are a strict academic evaluator. 
Given a MODEL ANSWER and a list of FACTUAL NUGGETS, your job is to determine whether the MODEL ANSWER explicitly contains the information described in each nugget.

MODEL ANSWER:
{model_answer}

FACTUAL NUGGETS:
{nuggets_text}

For each nugget, decide if its core information is covered in the model answer. It does not need to be an exact quote, but the factual meaning must be present.

Return ONLY a valid JSON object containing an array named "evaluations", like so:
{{
  "evaluations": [
    {{"id": 1, "covered": true}},
    {{"id": 2, "covered": false}}
  ]
}}
"""

def save_atomic(data, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=str, required=True, help="Path to the generated JSONL responses file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--provider", type=str, default="groq", choices=["gemini", "groq"], help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Model name to use under the selected provider")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        logger.error(f"Responses file not found: {responses_path}")
        sys.exit(1)

    nuggets_path = DATA_PROCESSED / "nuggets.json"
    if not nuggets_path.exists():
        logger.error(f"Nuggets file not found: {nuggets_path}")
        sys.exit(1)

    with open(nuggets_path, "r", encoding="utf-8") as f:
        all_nuggets = json.load(f)

    # Load responses
    responses = []
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))

    # Identify the answer key (qwen_answer or llama_answer)
    answer_key = None
    if len(responses) > 0:
        keys = list(responses[0].keys())
        for k in keys:
            if k.endswith("_answer"):
                answer_key = k
                break
    
    if not answer_key:
        logger.error("Could not find an answer key ending in '_answer' in the responses file.")
        sys.exit(1)

    logger.info(f"Using answer key: {answer_key}")

    output_name = responses_path.stem + "_eval.json"
    output_path = EVALS_DIR / output_name
    
    results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Resuming from checkpoint: {len(results)} evaluated")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    total_vital = 0
    covered_vital = 0
    total_okay = 0
    covered_okay = 0
    failed = 0

    processed = 0

    for idx, r in enumerate(responses):
        query_id = str(r["query_id"])
        
        if query_id in results:
            eval_data = results[query_id]
            total_vital += eval_data["total_vital"]
            covered_vital += eval_data["covered_vital"]
            total_okay += eval_data["total_okay"]
            covered_okay += eval_data["covered_okay"]
            continue

        if args.limit and processed >= args.limit:
            break

        if query_id not in all_nuggets:
            logger.warning(f"No nuggets found for query {query_id}")
            continue

        query_nuggets_data = all_nuggets[query_id]
        nuggets_list = query_nuggets_data["all_nuggets"]
        model_answer = r.get(answer_key, "").strip()

        if not nuggets_list:
            logger.warning(f"Empty nuggets list for query {query_id}")
            continue
            
        nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])

        prompt = PROMPT_TEMPLATE.format(
            model_answer=model_answer,
            nuggets_text=nuggets_text
        )

        success = False
        for attempt in range(3):
            try:
                if args.provider == "gemini":
                    resp_text = call_gemini_with_rate_limit(prompt, model_name=args.model)
                else:
                    resp_text = call_groq_with_rate_limit(prompt)
                    
                # Strip markdown code fences if present
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", resp_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()
                
                eval_json = json.loads(cleaned_text)
                evaluations = eval_json.get("evaluations", [])
                
                # Map evaluations
                eval_map = {item["id"]: item.get("covered", False) for item in evaluations}
                
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

                total_vital += query_vital
                covered_vital += query_covered_vital
                total_okay += query_okay
                covered_okay += query_covered_okay

                results[query_id] = {
                    "query_id": query_id,
                    "query_text": r.get("query_text", ""),
                    "total_vital": query_vital,
                    "covered_vital": query_covered_vital,
                    "vital_recall": (query_covered_vital / query_vital * 100) if query_vital > 0 else 0,
                    "total_okay": query_okay,
                    "covered_okay": query_covered_okay,
                    "okay_recall": (query_covered_okay / query_okay * 100) if query_okay > 0 else 0,
                    "nugget_results": nugget_results
                }
                
                save_atomic(results, output_path)
                logger.info(f"Query {query_id}: {query_covered_vital}/{query_vital} vital, {query_covered_okay}/{query_okay} okay")
                success = True
                break
            except Exception as e:
                logger.warning(f"Error on query {query_id} attempt {attempt+1}: {e}")
                if 'resp_text' in locals():
                    logger.warning(f"Raw response text:\n{resp_text}")
                time.sleep(10)

        if not success:
            logger.error(f"Failed to evaluate query {query_id}")
            failed += 1
            
        processed += 1

    vital_recall = (covered_vital / total_vital * 100) if total_vital > 0 else 0
    okay_recall = (covered_okay / total_okay * 100) if total_okay > 0 else 0

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Total Vital Nuggets: {covered_vital} / {total_vital} ({vital_recall:.1f}% Recall)")
    print(f"Total Okay Nuggets:  {covered_okay} / {total_okay} ({okay_recall:.1f}% Recall)")
    print(f"Total Failed Queries: {failed}")
    print(f"Detailed results saved to: {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
