"""
07_evaluate_support.py

Evaluates the Hallucination Index (Support Rate) for the generated answers using Gemini 1.5 Flash.
Classifies each statement in the answer as Fully Supported, Partially Supported, or Unsupported.
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
import argparse
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Literal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("07_evaluate_support")

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"
EVALS_DIR = DATA_PROCESSED / "evaluations"

# Load environment variables
load_dotenv(PIPELINE_ROOT.parent / ".env")
load_dotenv(PIPELINE_ROOT / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env")
    sys.exit(1)

client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=2, initial_delay=2.0)
    )
)

# Rate Limiter setup (15 RPM for Gemini free tier -> wait 4s per call)
MIN_INTERVAL = 60.0 / 14.0
last_call_time = 0.0

class StatementEvaluation(BaseModel):
    text: str
    support_level: Literal["fully_supported", "partially_supported", "unsupported"]

class SupportEvaluation(BaseModel):
    statements: list[StatementEvaluation]

def call_gemini_with_rate_limit(prompt: str, model_name: str = "gemini-3.1-flash-lite") -> str:
    global last_call_time
    elapsed = time.time() - last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    last_call_time = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2000,
            response_mime_type="application/json",
            response_schema=SupportEvaluation,
        ),
    )
    return response.text.strip()

PROMPT_TEMPLATE = """You are a strict fact-checker.
I will provide you with a MODEL ANSWER and a set of SOURCE DOCUMENTS.
Your job is to decompose the MODEL ANSWER into atomic statements/claims and classify the support level for EACH statement based ONLY on the SOURCE DOCUMENTS.

SOURCE DOCUMENTS:
{source_docs_text}

MODEL ANSWER:
{model_answer}

For each statement in the model answer, classify it into exactly one of these categories:
- "fully_supported": The statement is explicitly backed up by the provided documents.
- "partially_supported": The statement contains some truth from the documents, but extrapolates or alters details.
- "unsupported": The statement contains claims that cannot be found anywhere in the provided documents (hallucination).

Return ONLY a valid JSON object containing an array named "statements", like so:
{{
  "statements": [
    {{"text": "Apples are red.", "support_level": "fully_supported"}},
    {{"text": "They grow in winter.", "support_level": "unsupported"}}
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
    parser.add_argument("--contexts", type=str, required=True, help="Path to the context JSONL file (e.g. condition_1.jsonl)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Model name to use")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    contexts_path = Path(args.contexts)

    if not responses_path.exists():
        logger.error(f"Responses file not found: {responses_path}")
        sys.exit(1)
    if not contexts_path.exists():
        logger.error(f"Contexts file not found: {contexts_path}")
        sys.exit(1)

    # Load contexts
    contexts = {}
    with open(contexts_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                contexts[str(data["query_id"])] = data["context_docs"]

    # Load responses
    responses = []
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))

    # Identify the answer key
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

    output_name = responses_path.stem + "_support_eval.json"
    output_path = EVALS_DIR / output_name
    
    results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Resuming from checkpoint: {len(results)} evaluated")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    processed = 0
    total_fully = 0
    total_partially = 0
    total_unsupported = 0
    failed = 0

    for idx, r in enumerate(responses):
        query_id = str(r["query_id"])
        
        if query_id in results:
            eval_data = results[query_id]
            total_fully += eval_data["fully_supported"]
            total_partially += eval_data["partially_supported"]
            total_unsupported += eval_data["unsupported"]
            continue

        if args.limit and processed >= args.limit:
            break

        if query_id not in contexts:
            logger.warning(f"No context found for query {query_id}")
            continue

        context_docs = contexts[query_id]
        model_answer = r.get(answer_key, "").strip()
        
        if not model_answer:
            logger.warning(f"Empty model answer for query {query_id}")
            continue

        # Build context text
        passages_text_blocks = []
        for i, doc in enumerate(context_docs, start=1):
            title = doc.get("title", "")
            text_content = doc.get("text", "")
            passages_text_blocks.append(f"[Document {i}]\nTitle: {title}\n{text_content}")

        source_docs_text = "\n\n---\n\n".join(passages_text_blocks)

        prompt = PROMPT_TEMPLATE.format(
            source_docs_text=source_docs_text,
            model_answer=model_answer
        )

        success = False
        try:
            logger.info(f"Evaluating query {query_id} using model {args.model}...")
            resp_text = call_gemini_with_rate_limit(prompt, model_name=args.model)
            if not resp_text:
                raise ValueError("Empty response received")
            
            # Strip markdown code fences if present
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", resp_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()
            
            eval_json = json.loads(cleaned_text)
            statements = eval_json.get("statements", [])
            if not statements and len(model_answer) > 0:
                raise ValueError("No statements found in evaluation output")
            
            fully = 0
            partially = 0
            unsupported = 0
            
            for stmt in statements:
                lvl = stmt.get("support_level", "unsupported").lower()
                if lvl == "fully_supported":
                    fully += 1
                elif lvl == "partially_supported":
                    partially += 1
                else:
                    unsupported += 1

            total_statements = fully + partially + unsupported
            support_score = (fully + 0.5 * partially) / total_statements if total_statements > 0 else 0

            results[query_id] = {
                "query_id": query_id,
                "statements": statements,
                "fully_supported": fully,
                "partially_supported": partially,
                "unsupported": unsupported,
                "total_statements": total_statements,
                "support_score": support_score
            }
            
            save_atomic(results, output_path)
            
            total_fully += fully
            total_partially += partially
            total_unsupported += unsupported
            
            total_statements_all = total_fully + total_partially + total_unsupported
            running_support = ((total_fully + 0.5 * total_partially) / total_statements_all * 100) if total_statements_all > 0 else 0.0
            running_hallucination = (total_unsupported / total_statements_all * 100) if total_statements_all > 0 else 0.0
            progress_pct = ((idx + 1) / len(responses)) * 100
            
            logger.info(
                f"[{responses_path.name}] Progress: {idx+1}/{len(responses)} ({progress_pct:.1f}%) | "
                f"Query {query_id} SUCCEEDED: {fully} Fully, {partially} Partially, {unsupported} Unsupported | "
                f"Running Support: {running_support:.1f}%, Hallucination: {running_hallucination:.1f}%"
            )
            success = True
        except Exception as e:
            logger.error(f"Failed to evaluate query {query_id} using model {args.model}: {e}")
            failed += 1

        processed += 1

    total_statements_all = total_fully + total_partially + total_unsupported
    global_support_score = ((total_fully + 0.5 * total_partially) / total_statements_all * 100) if total_statements_all > 0 else 0
    hallucination_rate = (total_unsupported / total_statements_all * 100) if total_statements_all > 0 else 0

    print("\n" + "="*50)
    print("SUPPORT EVALUATION COMPLETE")
    print("="*50)
    print(f"Total Statements Analyzed: {total_statements_all}")
    print(f"Fully Supported:     {total_fully} ({total_fully/total_statements_all*100:.1f}%)")
    print(f"Partially Supported: {total_partially} ({total_partially/total_statements_all*100:.1f}%)")
    print(f"Unsupported:         {total_unsupported} ({total_unsupported/total_statements_all*100:.1f}%)")
    print(f"--> Global Support Score:  {global_support_score:.1f}%")
    print(f"--> Hallucination Rate:    {hallucination_rate:.1f}%")
    print(f"Total Failed Queries: {failed}")
    print(f"Detailed results saved to: {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
