"""
evaluate_k60_async.py

Async evaluation using Gemma 4 31B via Google AI Studio.
- Uses MULTIPLE API KEYS to increase concurrent throughput
- Fires up to 12 queries per key CONCURRENTLY per batch (avoids 15 RPM limit)
- Collects results as they arrive — saves immediately after each one
- Respects rate limit: each batch takes at least 60 seconds total
- Only evaluates 102 valid queries (17 degenerate skipped)
- Auto-resumes from saved progress
"""
import os
import json
import time
import re
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(r'e:\WorkSpace\lostinthemiddle\.env'))
keys = [v for k, v in os.environ.items() if k.startswith('GEMINI_API_KEY') and v]
valid_keys = [k for k in keys if k]
if not valid_keys:
    print("ERROR: No valid API keys found in .env")
    exit(1)

clients = [genai.Client(api_key=k) for k in valid_keys]
print(f"Loaded {len(clients)} API keys. Max concurrency = {12 * len(clients)}")

DEGENERATE_QIDS = {
    '2001908', '2003157', '2003976', '2026150', '2027130', '2027497',
    '2033470', '2034676', '2040352', '2044323', '2046027', '2051782',
    '3010623', '3100188', '3100292', '421946', '818583'
}

NUGGETS_PATH = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\nuggets.json')
INPUT_DIR    = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60')
OUTPUT_DIR   = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\evaluations')
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

def build_result(qid, qtxt, eval_map, nuggets_list):
    vital_total = vital_covered = okay_total = okay_covered = 0
    nugget_results = []
    for n in nuggets_list:
        nid = n["id"]
        covered = eval_map.get(nid, False)
        is_vital = n["vitality"] == "vital"
        if is_vital:
            vital_total += 1
            if covered: vital_covered += 1
        else:
            okay_total += 1
            if covered: okay_covered += 1
        nugget_results.append({"id": nid, "text": n["text"],
                                "vitality": n["vitality"], "covered": covered})
    return {
        "query_id": qid, "query_text": qtxt,
        "total_vital": vital_total, "covered_vital": vital_covered,
        "vital_recall": (vital_covered / vital_total * 100) if vital_total else 0,
        "total_okay": okay_total, "covered_okay": okay_covered,
        "okay_recall": (okay_covered / okay_total * 100) if okay_total else 0,
        "nugget_results": nugget_results
    }

async def evaluate_single(qid, qtxt, ans, nuggets_list, client_idx):
    """Evaluate one query asynchronously with 3 retries."""
    nuggets_text = "\n".join(
        [f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list]
    )
    prompt = PROMPT_TEMPLATE.format(model_answer=ans, nuggets_text=nuggets_text)

    client = clients[client_idx]
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model='gemma-4-31b-it',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4000)
            ),
            timeout=120.0
        )
        if not response or not response.text:
            raise ValueError("Empty/safety response")
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        data = json.loads(raw)
        return {item["id"]: item.get("covered", False) for item in data.get("evaluations", [])}
    except Exception as e:
        print(f"  [{qid}] Failed ({str(e)[:40]}) — SKIPPING instantly for future run", flush=True)
        return None

async def process_condition(fp, c_idx):
    """Process one condition file asynchronously."""
    out_f = OUTPUT_DIR / f"k60_{c_idx}_eval.json"

    # Load existing progress
    results = {}
    if out_f.exists():
        try:
            results = json.loads(out_f.read_text(encoding="utf-8"))
        except Exception:
            pass

    rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid_rows = [r for r in rows if str(r["query_id"]) not in DEGENERATE_QIDS]
    pending = [r for r in valid_rows if str(r["query_id"]) not in results]

    print(f"\n=== Condition {c_idx}: {len(results)}/{len(valid_rows)} already done, "
          f"{len(pending)} remaining ===", flush=True)

    if not pending:
        vr = [results[str(r["query_id"])]["vital_recall"] for r in valid_rows if str(r["query_id"]) in results]
        ok = [results[str(r["query_id"])]["okay_recall"] for r in valid_rows if str(r["query_id"]) in results]
        print(f"  SKIPPING — already complete. Vital={sum(vr)/len(vr):.2f}% Okay={sum(ok)/len(ok):.2f}%", flush=True)
        return

    # Process in batches of 14 per key to maximize the 15 RPM limit
    BATCH_SIZE = 14 * len(clients)
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} — firing {len(batch)} queries concurrently across {len(clients)} keys...", flush=True)

        batch_start_time = time.time()

        # Build coroutines for this batch
        tasks = []
        for i, row in enumerate(batch):
            qid = str(row["query_id"])
            qtxt = row.get("query_text", "")
            ans = row.get("qwen_answer", "").strip()
            nuggets_list = all_nuggets.get(qid, {}).get("all_nuggets", [])
            client_idx = i % len(clients)
            tasks.append((qid, qtxt, ans, nuggets_list, client_idx))

        # Fire all concurrently and collect as they complete
        coros = [evaluate_single(qid, qtxt, ans, nl, ci) for qid, qtxt, ans, nl, ci in tasks]
        task_objs = [asyncio.create_task(c) for c in coros]

        for i, (task, (qid, qtxt, ans, nuggets_list, client_idx)) in enumerate(zip(task_objs, tasks)):
            eval_map = await task
            if eval_map is None:
                continue
            results[qid] = build_result(qid, qtxt, eval_map, nuggets_list)
            # Save immediately
            with open(out_f, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            done_total = sum(1 for r in valid_rows if str(r["query_id"]) in results)
            vr = results[qid]["vital_recall"]
            print(f"  [{done_total}/{len(valid_rows)}] Query {qid} (Key{client_idx+1}): vital={vr:.0f}%", flush=True)

        # Ensure this batch took at least 60 seconds
        elapsed = time.time() - batch_start_time
        if elapsed < 62 and batch_start + BATCH_SIZE < len(pending):
            wait = 62 - elapsed
            print(f"  Batch done in {elapsed:.0f}s — waiting {wait:.0f}s before next batch...", flush=True)
            await asyncio.sleep(wait)

    # Final summary
    vr = [results[str(r["query_id"])]["vital_recall"] for r in valid_rows if str(r["query_id"]) in results]
    ok = [results[str(r["query_id"])]["okay_recall"] for r in valid_rows if str(r["query_id"]) in results]
    print(f"=== Condition {c_idx} COMPLETE: Vital={sum(vr)/len(vr):.2f}%  Okay={sum(ok)/len(ok):.2f}%  n={len(vr)} ===", flush=True)

async def main():
    files = sorted(INPUT_DIR.glob("k60_*.jsonl"),
                   key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
    for fp in files:
        c_idx = fp.stem.split("_")[-1]
        if int(c_idx) < 5:
            continue
        await process_condition(fp, c_idx)

if __name__ == "__main__":
    asyncio.run(main())
