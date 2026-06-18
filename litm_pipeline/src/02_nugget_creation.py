"""
02_nugget_creation.py

Extracts key information nuggets for each of the 119 research queries from
their golden documents using the Gemini Flash API, following the document selection,
prompting, rate-limiting, and validation rules specified.
"""

import os
import sys
import json
import time
import re
import string
import logging
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Force UTF-8 encoding for stdout and stderr (essential on Windows/certain shells)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("02_nugget_creation")

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"

# Load environment variables
load_dotenv(PIPELINE_ROOT.parent / ".env")
load_dotenv(PIPELINE_ROOT / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Gemini Client if key exists
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {e}")

# Rate Limiter setup
MIN_INTERVAL = 60.0 / 14  # 14 RPM safety margin
last_call_time = 0.0

def call_gemini_with_rate_limit(prompt: str) -> str:
    """Sends a request to Gemini Flash API, obeying the 14 RPM rate limit."""
    global last_call_time
    if not client:
        raise ValueError("Gemini client is not initialized. Check GEMINI_API_KEY.")
    elapsed = time.time() - last_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    last_call_time = time.time()

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1000,
            response_mime_type="application/json",
        ),
    )
    return response.text.strip()

# Groq Rate Limiter setup
last_groq_call_time = 0.0

def call_groq_with_rate_limit(prompt: str) -> str:
    """Sends a request to Groq API, obeying the 14 RPM rate limit."""
    global last_groq_call_time
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not found in .env file.")
    elapsed = time.time() - last_groq_call_time
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    last_groq_call_time = time.time()

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1000
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req) as response:
            resp_data = response.read().decode("utf-8")
            resp_json = json.loads(resp_data)
            return resp_json["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise RuntimeError(f"Groq API HTTP Error {e.code}: {error_body}")

def call_llm(prompt: str, provider: str) -> str:
    """Dispatches call to selected provider."""
    if provider == "groq":
        return call_groq_with_rate_limit(prompt)
    else:
        return call_gemini_with_rate_limit(prompt)

def normalize_text_for_dedup(text: str) -> str:
    """Normalizes text for duplicate detection: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text

def validate_and_clean_nuggets(nuggets: list) -> tuple[bool, str | list]:
    """Validates nugget structure, removes duplicate texts, and standardizes nugget IDs."""
    if not isinstance(nuggets, list):
        return False, "Response is not a list"

    cleaned = []
    seen_texts = set()

    for idx, n in enumerate(nuggets):
        if not isinstance(n, dict):
            return False, f"Nugget at index {idx} is not a dictionary"
        for key in ["id", "text", "vitality"]:
            if key not in n:
                return False, f"Nugget at index {idx} is missing key: '{key}'"
        if n["vitality"] not in ["vital", "okay"]:
            return False, f"Nugget at index {idx} has invalid vitality: '{n['vitality']}'"

        # Deduplication check
        norm_text = normalize_text_for_dedup(n["text"])
        if norm_text not in seen_texts:
            seen_texts.add(norm_text)
            cleaned.append(n)

    # Assign standard incremental IDs (1, 2, 3...)
    for i, n in enumerate(cleaned, start=1):
        n["id"] = i

    return True, cleaned

def get_title_and_text(passage: dict) -> tuple[str, str]:
    """Retrieves or extracts title and clean text from a passage dict."""
    title = passage.get("title")
    text = passage.get("text", "")

    if title:
        return title, text

    # If title is missing, try to extract from the first line of text
    lines = text.split("\n")
    if lines:
        extracted_title = lines[0].strip()
        # If the first line is reasonably short, assume it is the title
        if len(extracted_title) < 200:
            remaining_text = "\n".join(lines[1:]).strip()
            if remaining_text:
                return extracted_title, remaining_text
            else:
                return extracted_title, text

    return "Unknown Title", text

def save_atomic(data: dict, filepath: Path):
    """Saves data to a JSON file atomically using a temp file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, filepath)

# Direct exact prompt template from instructions
PROMPT_TEMPLATE = """You are a fact extraction specialist. Given a query and a set of
relevant documents, extract all key information nuggets that a
comprehensive answer to the query MUST cover.

A nugget is:
- One discrete, verifiable fact or argument (one sentence max)
- Grounded strictly in the provided documents — no outside knowledge
- Directly relevant to answering the query

For each nugget assign a vitality label:
- "vital": This fact is essential. Without it the answer is seriously incomplete.
- "okay": Useful supporting detail but the answer can still be good without it.

QUERY: {query_text}

SOURCE DOCUMENTS:
{source_docs_text}

Rules:
- Extract between 6 and 15 nuggets depending on how much information the documents contain
- Do NOT repeat the same fact in different words
- Do NOT invent facts not present in the documents
- Each nugget must be traceable to at least one source document
- Write each nugget as a complete, standalone sentence

Return ONLY a valid JSON array, no other text, no markdown fences:
[
  {{"id": 1, "text": "nugget text here", "vitality": "vital"}},
  {{"id": 2, "text": "nugget text here", "vitality": "okay"}}
]"""

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract nuggets using LLM API")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of new queries to process (for testing)")
    parser.add_argument("--provider", type=str, default="groq", choices=["gemini", "groq"], help="LLM provider to use (default: groq)")
    args = parser.parse_args()

    passages_path = DATA_PROCESSED / "per_query_passages.json"
    checkpoint_path = DATA_PROCESSED / "nuggets_checkpoint.json"
    output_path = DATA_PROCESSED / "nuggets.json"

    if not passages_path.exists():
        logger.error(f"Input file not found at {passages_path}. Run 01_data_preparation.py first.")
        sys.exit(1)

    with open(passages_path, "r", encoding="utf-8") as f:
        per_query_passages = json.load(f)

    # Checkpoint logic
    results = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming from checkpoint: {len(results)} queries already done")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Starting fresh.")

    total_queries = len(per_query_passages)
    failed_queries = 0
    processed_this_run = 0
    consecutive_failures = 0

    for idx, (query_id, data) in enumerate(per_query_passages.items(), start=1):
        if query_id in results:
            continue

        if args.limit is not None and processed_this_run >= args.limit:
            break

        query_text = data["query_text"]
        golden_passages = data.get("golden_passages", [])

        # Document Selection Logic
        # Step 1: Separate golden_passages into two groups
        qrel3_docs = [p for p in golden_passages if p.get("qrel") == 3]
        qrel2_docs = [p for p in golden_passages if p.get("qrel") == 2]

        # Step 2: Sort both by bm25_score descending
        qrel3_docs_sorted = sorted(qrel3_docs, key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        qrel2_docs_sorted = sorted(qrel2_docs, key=lambda x: x.get("bm25_score", 0.0), reverse=True)

        # Step 3 & 4: Combine and trim to max 20 total docs, prioritizing qrel=3
        source_docs = (qrel3_docs_sorted + qrel2_docs_sorted)[:20]
        
        # Calculate how many of each we are using
        n_qrel3_used = sum(1 for p in source_docs if p.get("qrel") == 3)
        n_qrel2_used = sum(1 for p in source_docs if p.get("qrel") == 2)
        total_source_docs = len(source_docs)

        # Log document details for this query
        logger.info(f"Query {query_id}: {n_qrel3_used} qrel=3 docs + {n_qrel2_used} qrel=2 docs = {total_source_docs} source docs")

        # Step 5: Handle warning / defensive skip
        if total_source_docs < 3:
            logger.warning(f"Skipping query {query_id}: only {total_source_docs} source docs (minimum 3 required).")
            continue

        # Build source docs text block
        passages_text_blocks = []
        for i, p in enumerate(source_docs, start=1):
            title, text_content = get_title_and_text(p)
            passages_text_blocks.append(f"[Source Doc {i}] Title: {title}\n{text_content}")

        source_docs_text = "\n\n---\n\n".join(passages_text_blocks)

        # Format the prompt
        prompt = PROMPT_TEMPLATE.format(
            query_text=query_text,
            source_docs_text=source_docs_text
        )

        # API Call & Validation loop
        success = False
        cleaned_nuggets = None

        for attempt in range(2):
            try:
                result_text = call_llm(prompt, args.provider)

                # Strip markdown code fences if present
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", result_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

                # Parse JSON
                nuggets = json.loads(cleaned_text)

                # Validate and clean
                valid, cleaned = validate_and_clean_nuggets(nuggets)
                if valid:
                    cleaned_nuggets = cleaned
                    success = True
                    break
                else:
                    logger.warning(f"Validation failed for query {query_id} on attempt {attempt+1}: {cleaned}")
                    if attempt < 1:
                        wait_time = 10 * (attempt + 1)
                        logger.info(f"Sleeping for {wait_time}s before retrying...")
                        time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"API/Parsing error on attempt {attempt+1} for query {query_id}: {e}")
                if attempt < 1:
                    wait_time = 20 if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else 5
                    logger.info(f"Sleeping for {wait_time}s before retrying...")
                    time.sleep(wait_time)

        if not success:
            logger.error(f"FAILED: query {query_id}")
            failed_queries += 1
            consecutive_failures += 1
            if consecutive_failures >= 5:
                logger.critical("5 consecutive failures detected. LLM quota might be exhausted. Exiting loop.")
                break
            continue

        # Log a warning if nugget count is not in 6-15, but keep it
        if not (6 <= len(cleaned_nuggets) <= 15):
            logger.warning(f"Query {query_id} generated {len(cleaned_nuggets)} nuggets (expected between 6 and 15). Keeping anyway.")

        # Separate into vital and okay
        vital_nuggets = [n for n in cleaned_nuggets if n["vitality"] == "vital"]
        okay_nuggets = [n for n in cleaned_nuggets if n["vitality"] == "okay"]

        # Package output
        query_result = {
            "query_text": query_text,
            "n_source_docs_used": total_source_docs,
            "n_qrel3_used": n_qrel3_used,
            "n_qrel2_used": n_qrel2_used,
            "all_nuggets": cleaned_nuggets,
            "vital_nuggets": vital_nuggets,
            "okay_nuggets": okay_nuggets,
            "total_count": len(cleaned_nuggets),
            "vital_count": len(vital_nuggets),
            "okay_count": len(okay_nuggets)
        }

        # Save query result
        results[query_id] = query_result

        # Save checkpoint immediately
        save_atomic(results, checkpoint_path)

        consecutive_failures = 0

        # Print progress to console
        print(f"  [{idx}/{total_queries}] Query {query_id} → {total_source_docs} source docs → {len(cleaned_nuggets)} nuggets ({len(vital_nuggets)} vital, {len(okay_nuggets)} okay)")

        processed_this_run += 1

    # Save final output
    save_atomic(results, output_path)

    # Calculate statistics
    successful_results = [res for res in results.values()]
    success_count = len(successful_results)

    total_nuggets = sum(res["total_count"] for res in successful_results)
    total_vital = sum(res["vital_count"] for res in successful_results)

    mean_nuggets = total_nuggets / success_count if success_count > 0 else 0.0
    mean_vital = total_vital / success_count if success_count > 0 else 0.0
    min_nuggets = min(res["total_count"] for res in successful_results) if success_count > 0 else 0
    max_nuggets = max(res["total_count"] for res in successful_results) if success_count > 0 else 0

    print("\n══════════════════════════════════════")
    print("NUGGET CREATION COMPLETE")
    print("══════════════════════════════════════")
    print(f"Total queries processed:  {total_queries}")
    print(f"Total queries failed:     {failed_queries}")
    print(f"Total nuggets created:    {total_nuggets:,}")
    print(f"Mean nuggets per query:   {mean_nuggets:.1f}")
    print(f"Mean vital per query:     {mean_vital:.1f}")
    print(f"Min nuggets (any query):  {min_nuggets}")
    print(f"Max nuggets (any query):  {max_nuggets}")
    print(f"Output saved to: {output_path.relative_to(PIPELINE_ROOT) if output_path.is_relative_to(PIPELINE_ROOT) else output_path}")
    print("══════════════════════════════════════")

if __name__ == "__main__":
    main()
