import os, json, time, re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(r'e:\WorkSpace\lostinthemiddle\.env'))
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

nuggets_path = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\data\processed\nuggets.json')
all_nuggets = json.loads(nuggets_path.read_text(encoding='utf-8'))
fp = Path(r'e:\WorkSpace\lostinthemiddle\litm_pipeline\results\answers\k=60\k60_1.jsonl')
rows = [json.loads(l) for l in fp.read_text(encoding='utf-8').splitlines() if l.strip()]
r = next(x for x in rows if str(x['query_id']) == '23287')

ans = r['qwen_answer'].strip()
nuggets_list = all_nuggets['23287']['all_nuggets']
nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])
prompt = f"MODEL ANSWER:\n{ans}\n\nFACTUAL NUGGETS:\n{nuggets_text}\n\nOutput compact single-line JSON only:\n{{\"evaluations\":[{{\"id\":1,\"covered\":true}}]}}"

print(f"Sending prompt ({len(prompt)} chars, ~{len(prompt)//4} tokens)...", flush=True)
start = time.time()
try:
    response = client.models.generate_content(
        model='gemma-4-31b-it',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4000)
    )
    elapsed = time.time() - start
    print(f"SUCCESS in {elapsed:.1f}s")
    print(f"Response: {response.text[:300]}")
except Exception as e:
    print(f"ERROR in {time.time()-start:.1f}s: {e}")
