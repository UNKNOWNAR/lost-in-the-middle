"""
06_evaluate_answers_kaggle_gemma.py

Evaluates generated answers (JSONL format) against ground-truth nuggets.
Uses vLLM with a local Gemma model on Kaggle GPUs (supports T4 x2 tensor parallelism).
Uses batch generation for high throughput.
"""

import os
import sys
import json
import time
import re
import logging
import argparse
from pathlib import Path
from pydantic import BaseModel

# Setup environment for Kaggle T4 x2
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("06_evaluate_answers_kaggle_gemma")

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

def save_atomic(data, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, filepath)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=str, required=True, help="Path to the generated JSONL responses file")
    parser.add_argument("--model", type=str, default="/kaggle/input/gemma-2/transformers/9b-it/2", help="Model name or local path")
    parser.add_argument("--nuggets", type=str, default=None, help="Path to nuggets.json. Auto-detected if not set.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Force tensor parallel size")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        logger.error(f"Responses file not found: {responses_path}")
        sys.exit(1)

    # Resolve nuggets path
    if args.nuggets:
        nuggets_path = Path(args.nuggets)
    else:
        # Default Kaggle dataset or local path
        if os.path.exists("/kaggle/input"):
            # Check the specific uploaded nuggets path first
            specific_path = Path("/kaggle/input/datasets/arinjaysarkar/nuggets/nuggets.json")
            if specific_path.exists():
                nuggets_path = specific_path
            else:
                # Search under /kaggle/input/ for nuggets.json
                nuggets_search = list(Path("/kaggle/input").glob("**/nuggets.json"))
                if nuggets_search:
                    nuggets_path = nuggets_search[0]
                else:
                    nuggets_path = Path("/kaggle/input/datasets/arinjaysarkar/k40data/nuggets.json")
        else:
            nuggets_path = responses_path.parent.parent / "nuggets.json"

    if not nuggets_path.exists():
        logger.error(f"Nuggets file not found: {nuggets_path}")
        sys.exit(1)

    logger.info(f"Loading nuggets from: {nuggets_path}")
    with open(nuggets_path, "r", encoding="utf-8") as f:
        all_nuggets = json.load(f)

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

    logger.info(f"Using answer key: {answer_key}")

    # Set up output path
    # On Kaggle, output to /kaggle/working/evaluations/
    if os.path.exists("/kaggle/input"):
        evals_dir = Path("/kaggle/working/evaluations")
    else:
        evals_dir = responses_path.parent.parent / "evaluations"
    
    output_name = responses_path.stem + "_eval.json"
    output_path = evals_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Resuming from checkpoint: {len(results)} evaluated")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    # Filter responses to evaluate
    to_evaluate = []
    for r in responses:
        qid = str(r["query_id"])
        if qid in results:
            continue
        to_evaluate.append(r)

    if not to_evaluate:
        logger.info("All responses already evaluated!")
        sys.exit(0)

    logger.info(f"Preparing to evaluate {len(to_evaluate)} items using local vLLM...")

    # Initialize vLLM
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    import torch

    ngpu = torch.cuda.device_count()
    
    model_path = Path(args.model)
    if not model_path.exists():
        detected_path = find_gemma_model_path()
        if detected_path:
            model_path = detected_path
            logger.info(f"Provided model path not found. Auto-detected Gemma model at: {model_path}")
        else:
            logger.error(f"Model path {args.model} not found and could not auto-detect any Gemma model under /kaggle/input.")
            sys.exit(1)
    else:
        logger.info(f"Using provided model path: {model_path}")

    is_27b = "27b" in str(model_path).lower()

    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else (min(ngpu, 2) if ngpu > 0 else 1)
    pp = 1

    # For bitsandbytes, tensor parallelism is not supported in vLLM.
    # We must use pipeline parallelism to split across multiple GPUs.
    if is_27b and ngpu > 1 and args.tensor_parallel_size is None:
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
        "max_model_len": 8192,
        "trust_remote_code": True,
        "dtype": "float16",
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "max_num_seqs": 16, # Prevent KV cache thrashing
    }
    
    if is_27b:
        logger.info("Enabling bitsandbytes 4-bit quantization to prevent OOM on 2x T4 GPUs (30GB total VRAM).")
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
        llm_kwargs["max_num_seqs"] = 8 # Limit concurrency further for quantized 27b

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = []
    valid_queries = []
    for r in to_evaluate:
        qid = str(r["query_id"])
        if qid not in all_nuggets:
            logger.warning(f"No nuggets found for query {qid}")
            continue
        
        query_nuggets_data = all_nuggets[qid]
        nuggets_list = query_nuggets_data["all_nuggets"]
        model_answer = r.get(answer_key, "").strip()

        if not nuggets_list:
            logger.warning(f"Empty nuggets list for query {qid}")
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
        logger.info("No valid prompts constructed for evaluation.")
        sys.exit(0)

    # Note: Guided decoding (outlines) is disabled here because FSM vocabulary masking 
    # for Gemma-2's 256k vocabulary on CPU has extremely high overhead, and can force 
    # the model to output empty strings. Gemma-2-27B is highly capable and follows JSON 
    # formatting instructions natively without constraints.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1500,
    )

    logger.info(f"Running batch generation for {len(prompts)} prompts...")
    t_start = time.time()
    outputs = llm.generate(prompts, params)
    logger.info(f"Generation completed in {time.time() - t_start:.2f}s")

    # Parse and record results
    total_vital = 0
    covered_vital = 0
    total_okay = 0
    covered_okay = 0
    failed = 0

    # Include existing metrics if resuming
    for qid, res in results.items():
        total_vital += res.get("total_vital", 0)
        covered_vital += res.get("covered_vital", 0)
        total_okay += res.get("total_okay", 0)
        covered_okay += res.get("covered_okay", 0)

    for (r, nuggets_list), out in zip(valid_queries, outputs):
        query_id = str(r["query_id"])
        resp_text = out.outputs[0].text.strip()

        try:
            # Clean markdown code block wraps if present
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
            logger.error(f"Failed to parse model output for query {query_id}: {e}")
            logger.debug(f"Raw output: {resp_text}")
            failed += 1

    save_atomic(results, output_path)

    vital_recall = (covered_vital / total_vital * 100) if total_vital > 0 else 0
    okay_recall = (covered_okay / total_okay * 100) if total_okay > 0 else 0

    print("\n" + "="*50)
    print("LOCAL BATCH EVALUATION COMPLETE")
    print("="*50)
    print(f"Total Vital Nuggets: {covered_vital} / {total_vital} ({vital_recall:.1f}% Recall)")
    print(f"Total Okay Nuggets:  {covered_okay} / {total_okay} ({okay_recall:.1f}% Recall)")
    print(f"Total Failed Queries: {failed}")
    print(f"Detailed results saved to: {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
