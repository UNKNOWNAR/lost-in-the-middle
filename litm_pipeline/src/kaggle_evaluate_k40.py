"""
kaggle_evaluate_k40.py

Self-contained script to run nugget evaluation for ALL k40_*.jsonl files (1 to 10)
using gemma-4-31b-it via vLLM on Kaggle T4 x2.
"""

import os
import sys
import json
import time
import re
import logging
from pathlib import Path
from pydantic import BaseModel

# Force vLLM configurations for Kaggle T4 GPUs
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("kaggle_eval_k40")

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

Return ONLY a valid JSON object containing an array named "evaluations", like so:
{{
  "evaluations": [
    {{"id": 1, "covered": true}},
    {{"id": 2, "covered": false}}
  ]
}}
"""

INPUT_DIR = Path("/kaggle/input/datasets/arinjaysarkar/gemma-answers/generated_answers/qwen")
OUTPUT_DIR = Path("/kaggle/working/evaluations")

def find_gemma_model_path():
    # Explicit search list for Gemma-2 in Kaggle environment
    search_paths = [
        Path("/kaggle/input/gemma-2/transformers/gemma-2-9b-it/2"),
        Path("/kaggle/input/gemma-2/transformers/gemma-2-9b-it/1"),
        Path("/kaggle/input/gemma-2/transformers/gemma-2-27b-it/2"),
        Path("/kaggle/input/gemma-2/transformers/gemma-2-27b-it/1"),
        Path("/kaggle/input/gemma-2/transformers/9b-it/2"),
        Path("/kaggle/input/gemma-2/transformers/27b-it/2"),
    ]
    for p in search_paths:
        if p.exists() and (p / "config.json").exists():
            return p

    # Fallback recursive search under /kaggle/input
    if os.path.exists("/kaggle/input"):
        logger.info("Searching recursively for Gemma model path under /kaggle/input ...")
        for root, dirs, files in os.walk("/kaggle/input"):
            if "config.json" in files and "gemma" in root.lower() and "transformers" in root.lower():
                # Avoid picking up gemma-4/gemma4 which is unsupported
                if "gemma-4" in root.lower() or "gemma4" in root.lower():
                    continue
                return Path(root)
    return None

def find_nuggets_file():
    # Explicit search list
    search_paths = [
        Path("/kaggle/input/datasets/arinjaysarkar/nuggets/nuggets.json"),
        Path("/kaggle/input/datasets/arinjaysarkar/k40data/nuggets.json"),
        Path("/kaggle/input/datasets/arinjaysarkar/litm-dataset/nuggets.json"),
        Path("/kaggle/input/datasets/arinjaysarkar/gemma-answers/nuggets.json"),
        Path("/kaggle/input/k40data/nuggets.json"),
        Path("/kaggle/input/gemma-answers/nuggets.json"),
        Path("/kaggle/input/nuggets.json"),
        Path("/kaggle/working/nuggets.json"),
        Path("./nuggets.json")
    ]
    for p in search_paths:
        if p.exists():
            return p

    # Fallback recursive search under /kaggle/input
    if os.path.exists("/kaggle/input"):
        logger.info("Searching recursively for nuggets.json under /kaggle/input ...")
        for root, dirs, files in os.walk("/kaggle/input"):
            if "nuggets.json" in files:
                return Path(root) / "nuggets.json"
                
    return None

def main():
    # Find nuggets file first
    nuggets_file = find_nuggets_file()
    if not nuggets_file or not nuggets_file.exists():
        logger.error("nuggets.json file not found anywhere under /kaggle/input/ or /kaggle/working/.")
        if os.path.exists("/kaggle/input"):
            logger.info("Contents of /kaggle/input:")
            for root, dirs, files in os.walk("/kaggle/input"):
                level = root.replace("/kaggle/input", "").count(os.sep)
                indent = " " * 4 * (level)
                logger.info(f"{indent}{os.path.basename(root)}/ : {files}")
        sys.exit(1)
        
    logger.info(f"Using Nuggets File: {nuggets_file}")
    with open(nuggets_file, "r", encoding="utf-8") as f:
        all_nuggets = json.load(f)

    # Find all k40_*.jsonl files to process
    if not INPUT_DIR.exists():
        logger.error(f"Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)
        
    input_files = sorted(list(INPUT_DIR.glob("k40_*.jsonl")), key=lambda x: int(re.findall(r"\d+", x.name)[-1]))
    if not input_files:
        logger.error(f"No k40_*.jsonl files found in {INPUT_DIR}")
        sys.exit(1)
        
    logger.info(f"Found {len(input_files)} files to evaluate: {[f.name for f in input_files]}")

    # Initialize vLLM
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    import torch

    ngpu = torch.cuda.device_count()
    model_path = find_gemma_model_path()
    if not model_path:
        logger.error("Could not find a valid Gemma model directory under /kaggle/input. Please ensure you have attached Gemma 2 in your notebook.")
        sys.exit(1)
        
    logger.info(f"Using Model Path: {model_path}")
    is_27b = "27b" in str(model_path).lower()

    tp = min(ngpu, 2) if ngpu > 0 else 1
    pp = 1

    # For bitsandbytes, tensor parallelism is not supported in vLLM.
    # We must use pipeline parallelism to split across multiple GPUs.
    if is_27b and ngpu > 1:
        logger.info(f"Detected Gemma-2-27B. Using pipeline parallelism (pipeline_parallel_size={ngpu}) instead of tensor parallelism to support bitsandbytes multi-GPU execution.")
        tp = 1
        pp = ngpu
    else:
        logger.info(f"Initializing vLLM with tensor_parallel_size={tp} and pipeline_parallel_size={pp}")

    # Build initialization parameters dynamically
    llm_kwargs = {
        "model": str(model_path),
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
        "max_model_len": 4096,
        "trust_remote_code": True,
        "dtype": "float16",
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 16, # Prevent KV cache thrashing by limiting parallel sequences
    }
    
    if is_27b:
        logger.info("Enabling bitsandbytes 4-bit quantization to prevent OOM on 2x T4 GPUs (30GB total VRAM).")
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
        llm_kwargs["max_num_seqs"] = 8 # Limit concurrency further for quantized 27B due to small KV cache headroom
        
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Note: Guided decoding (outlines) is disabled here because FSM vocabulary masking 
    # for Gemma-2's 256k vocabulary on CPU has extremely high overhead, and can force 
    # the model to output empty strings. Gemma-2-27B is highly capable and follows JSON 
    # formatting instructions natively without constraints.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1000,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, fp in enumerate(input_files, start=1):
        cond_num = re.findall(r"\d+", fp.name)[-1]
        out_f = OUTPUT_DIR / f"k40_{cond_num}_eval.json"
        
        if out_f.exists():
            logger.info(f"Output already exists for {fp.name}. Skipping.")
            continue
            
        logger.info(f"[{idx}/{len(input_files)}] Evaluating {fp.name} -> {out_f.name}")

        # Load responses for this file
        responses = []
        with open(fp, "r", encoding="utf-8") as f:
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
            answer_key = "qwen_answer"

        # Build prompts for batch
        prompts = []
        valid_queries = []
        for r in responses:
            qid = str(r["query_id"])
            if qid not in all_nuggets:
                continue
            
            query_nuggets_data = all_nuggets[qid]
            nuggets_list = query_nuggets_data["all_nuggets"]
            model_answer = r.get(answer_key, "").strip()

            if not nuggets_list or not model_answer:
                continue

            nuggets_text = "\n".join([f"ID {n['id']} (Vitality: {n['vitality']}): {n['text']}" for n in nuggets_list])
            prompt_text = PROMPT_TEMPLATE.format(
                model_answer=model_answer,
                nuggets_text=nuggets_text
            )
            
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
            valid_queries.append((r, nuggets_list))

        if not prompts:
            logger.warning(f"No prompts could be generated for {fp.name}. Skipping.")
            continue

        # Batch generation
        t_gen_start = time.time()
        outputs = llm.generate(prompts, params)
        logger.info(f"Generated {len(prompts)} evaluations in {time.time()-t_gen_start:.1f}s")

        results = {}
        total_vital = 0
        covered_vital = 0
        total_okay = 0
        covered_okay = 0
        failed = 0

        for (r, nuggets_list), out in zip(valid_queries, outputs):
            query_id = str(r["query_id"])
            resp_text = out.outputs[0].text.strip()

            try:
                cleaned_text = re.sub(r"^```(?:json)?\s*", "", resp_text)
                cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()
                
                eval_json = json.loads(cleaned_text)
                evaluations = eval_json.get("evaluations", [])
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
            except Exception as e:
                failed += 1
                logger.error(f"Failed to parse output for query {query_id}: {e}")
                logger.error(f"Raw output was: {resp_text!r}")

        # Write output file
        with open(out_f, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        vital_recall = (covered_vital / total_vital * 100) if total_vital > 0 else 0
        okay_recall = (covered_okay / total_okay * 100) if total_okay > 0 else 0
        logger.info(f"Finished {fp.name}: Vital Recall = {vital_recall:.1f}%, Okay Recall = {okay_recall:.1f}%, Failed = {failed}")

    print("\n" + "="*50)
    print("ALL EVALUATIONS COMPLETE!")
    print(f"Results saved under: {OUTPUT_DIR}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
