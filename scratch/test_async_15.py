"""
Test: Fire first 15 queries of condition 1 concurrently and see results arrive in real time.
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
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

DEGENERATE_QIDS = {
    '2001908', '2003157', '2003976', '2026150', '2027130', '2027497',
    '2033470', '2034676', '2040352', '2044323', '2046027', '2051782',
    '3010623', '3100188', '3100292', '421946', '818583'
}

all_nuggets = json.loads(Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\nuggets.json').read_text(encoding='utf-8'))

PROMPT_TEMPLATE = """You are a strict academic evaluator.
Given a MODEL ANSWER and a list of FACTUAL NUGGETS, determine whether the MODEL ANSWER contains the information in each nugget.

MODEL ANSWER:
{model_answer}

FACTUAL NUGGETS:
{nuggets_text}

Output compact single-line JSON only (no markdown, no newlines inside JSON, no explanation):
{{"evaluations":[{{"id":1,"covered":true}},{{"id":2,"covered":false}}]}}
"""

async def evaluate_single(qid, qtxt, ans, nuggets_list, idx):
    t0 = time.time()
    nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])
    prompt = PROMPT_TEMPLATE.format(model_answer=ans, nuggets_text=nuggets_text)
    backoff = [15, 30, 60]
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model='gemma-4-31b-it',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4000)
            )
            if not response or not response.text:
                raise ValueError("Empty response")
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
            data = json.loads(raw)
            eval_map = {item["id"]: item.get("covered", False) for item in data.get("evaluations", [])}
            elapsed = time.time() - t0
            vital_covered = sum(1 for n in nuggets_list if n['vitality']=='vital' and eval_map.get(n['id'], False))
            vital_total   = sum(1 for n in nuggets_list if n['vitality']=='vital')
            print(f"  [{idx:02d}] Query {qid} -> {vital_covered}/{vital_total} vital ({elapsed:.1f}s)", flush=True)
            return qid, eval_map
        except Exception as e:
            wait = backoff[attempt]
            print(f"  [{idx:02d}] Query {qid} attempt {attempt+1} failed: {str(e)[:60]} -> wait {wait}s", flush=True)
            await asyncio.sleep(wait)
    print(f"  [{idx:02d}] Query {qid} ALL RETRIES FAILED", flush=True)
    return qid, {}

async def main():
    fp = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60\k60_1.jsonl')
    rows = [json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines() if l.strip()]
    valid = [r for r in rows if str(r['query_id']) not in DEGENERATE_QIDS]
    batch = valid[:15]

    print(f"Firing {len(batch)} queries concurrently against gemma-4-31b-it...", flush=True)
    print(f"Start time: {time.strftime('%H:%M:%S')}", flush=True)

    t_start = time.time()
    tasks = [
        asyncio.create_task(evaluate_single(
            str(r['query_id']),
            r.get('query_text',''),
            r.get('qwen_answer','').strip(),
            all_nuggets.get(str(r['query_id']),{}).get('all_nuggets',[]),
            i+1
        ))
        for i, r in enumerate(batch)
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t_start

    successes = sum(1 for _, em in results if em)
    print(f"\nDone! {successes}/{len(batch)} succeeded in {elapsed:.1f}s total", flush=True)
    print(f"End time: {time.strftime('%H:%M:%S')}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
