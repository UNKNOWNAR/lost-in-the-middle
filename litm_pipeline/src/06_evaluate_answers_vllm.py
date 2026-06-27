"""
06_evaluate_answers_vllm.py

Evaluates generated answers (JSONL format) against the ground-truth nuggets.
Uses a local vLLM instance (Qwen 2.5 7B Instruct) on Kaggle to evaluate all conditions at scale.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from pydantic import BaseModel

# Setup logging (configured for Jupyter compatibility)
logger = logging.getLogger("06_evaluate_answers_vllm")
logger.setLevel(logging.INFO)
logger.handlers = []
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(ch)

# Model & Data Paths
MODEL_NAME = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"

# Search for the dataset under possible Kaggle mount paths
possible_data_bases = [
    Path("/kaggle/input/datasets/arinjaysarkar/data-k100"),
    Path("/kaggle/input/data-k100"),
]
DATA_BASE = next((p for p in possible_data_bases if p.exists()), possible_data_bases[0])

# Search for nuggets under possible Kaggle mount paths
possible_nugget_paths = [
    Path("/kaggle/input/datasets/arinjaysarkar/nuggets/nuggets.json"),
    Path("/kaggle/input/nuggets/nuggets.json"),
]
NUGGETS_PATH = next((p for p in possible_nugget_paths if p.exists()), possible_nugget_paths[0])

OUTPUT_DIR = Path("/kaggle/working/evaluations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pydantic Schema for Guided Decoding ──────────────────────────────────────
class NuggetEvaluation(BaseModel):
    id: int
    covered: bool

class AnswerEvaluation(BaseModel):
    evaluations: list[NuggetEvaluation]

PROMPT_TEMPLATE = """You are a strict academic evaluator. 
Given a MODEL ANSWER and a list of FACTUAL NUGGETS, your job is to determine whether the MODEL ANSWER explicitly contains the information described in each nugget.

MODEL ANSWER:
{model_answer}

FACTUAL NUGGETS:
{nuggets_text}

For each nugget, decide if its core information is covered in the model answer. It does not need to be an exact quote, but the factual meaning must be present.
"""

def main():
    if not NUGGETS_PATH.exists():
        logger.error(f"Nuggets file not found: {NUGGETS_PATH}")
        sys.exit(1)

    with open(NUGGETS_PATH, "r", encoding="utf-8") as f:
        all_nuggets = json.load(f)

    # 1. Initialize vLLM
    logger.info("Initializing vLLM model...")
    from vllm import LLM, SamplingParams
    import torch

    ngpu = torch.cuda.device_count()
    tp = min(ngpu, 2) if ngpu > 0 else 1

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tp,
        max_model_len=8192,  # Evaluation prompts are small (~2k tokens max)
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    tokenizer = llm.get_tokenizer()
    
    # Enable guided JSON decoding using Pydantic schema
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1500,
        guided_decoding={"json": AnswerEvaluation.model_json_schema()}
    )

    # Find all response files (k100_1.jsonl to k100_10.jsonl)
    input_files = sorted(DATA_BASE.glob("k100_*.jsonl"), key=lambda x: int(x.stem.split("_")[-1]))
    
    if not input_files:
        logger.error(f"No response files found in {DATA_BASE}")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} files to evaluate: {[f.name for f in input_files]}")

    for fp in input_files:
        c_idx = fp.stem.split("_")[-1]
        out_f = OUTPUT_DIR / f"k100_{c_idx}_eval.json"
        
        if out_f.exists():
            logger.info(f"Evaluation for condition {c_idx} already exists. Skipping.")
            continue

        logger.info(f"Evaluating {fp.name}...")
        t0 = time.time()
        
        # Read file
        rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        
        # Find answer key (qwen_answer or similar)
        answer_key = None
        if rows:
            for k in rows[0].keys():
                if k.endswith("_answer"):
                    answer_key = k
                    break
        if not answer_key:
            logger.error(f"Could not find answer key ending in '_answer' in {fp.name}")
            continue

        prompts = []
        queries_meta = []

        for d in rows:
            query_id = str(d["query_id"])
            query_text = d.get("query_text", "")
            model_answer = d.get(answer_key, "").strip()

            if query_id not in all_nuggets:
                logger.warning(f"No nuggets found for query {query_id}. Skipping.")
                continue

            query_nuggets_data = all_nuggets[query_id]
            nuggets_list = query_nuggets_data["all_nuggets"]

            if not nuggets_list:
                continue

            # Format prompt
            nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])
            raw_prompt = PROMPT_TEMPLATE.format(
                model_answer=model_answer,
                nuggets_text=nuggets_text
            )

            # Build chat messages
            msgs = [{"role": "user", "content": raw_prompt}]
            prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            
            prompts.append(prompt_str)
            queries_meta.append((query_id, query_text, nuggets_list))

        if not prompts:
            logger.warning(f"No prompts built for {fp.name}")
            continue

        # Run vLLM generation
        logger.info(f"Generating evaluations for {len(prompts)} queries in batch...")
        outputs = llm.generate(prompts, params)

        # Parse outputs and compile results
        results = {}
        for (query_id, query_text, nuggets_list), out in zip(queries_meta, outputs):
            try:
                eval_json = json.loads(out.outputs[0].text.strip())
                evaluations = eval_json.get("evaluations", [])
            except Exception as e:
                logger.error(f"Failed to parse model output for query {query_id}: {e}")
                evaluations = []

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

            results[query_id] = {
                "query_id": query_id,
                "query_text": query_text,
                "total_vital": query_vital,
                "covered_vital": query_covered_vital,
                "vital_recall": (query_covered_vital / query_vital * 100) if query_vital > 0 else 0,
                "total_okay": query_okay,
                "covered_okay": query_covered_okay,
                "okay_recall": (query_covered_okay / query_okay * 100) if query_okay > 0 else 0,
                "nugget_results": nugget_results
            }

        # Write results
        tmp_f = out_f.with_suffix(".tmp")
        with tmp_f.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        tmp_f.rename(out_f)

        logger.info(f"Finished {fp.name} in {time.time()-t0:.1f}s.")

    logger.info("All evaluations complete!")

if __name__ == "__main__":
    main()
