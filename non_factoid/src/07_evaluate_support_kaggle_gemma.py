"""
07_evaluate_support_kaggle_gemma.py

Evaluates the Hallucination Index (Support Rate) for generated answers.
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
from typing import Literal

# Setup environment for Kaggle T4 x2
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("07_evaluate_support_kaggle_gemma")

class StatementEvaluation(BaseModel):
    text: str
    support_level: Literal["fully_supported", "partially_supported", "unsupported"]

class SupportEvaluation(BaseModel):
    statements: list[StatementEvaluation]

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
    parser.add_argument("--contexts", type=str, required=True, help="Path to the context JSONL file (e.g. condition_1.jsonl)")
    parser.add_argument("--model", type=str, default="/kaggle/input/gemma-2/transformers/9b-it/2", help="Model name or local path")
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Force tensor parallel size")
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

    # Set up output path
    if os.path.exists("/kaggle/input"):
        evals_dir = Path("/kaggle/working/evaluations")
    else:
        evals_dir = responses_path.parent.parent / "evaluations"

    output_name = responses_path.stem + "_support_eval.json"
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
        "max_model_len": 16384, # Set higher limit for source documents contexts
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
        llm_kwargs["max_num_seqs"] = 4 # Limit concurrency even further for 16k max_model_len on 27b

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = []
    valid_queries = []
    for r in to_evaluate:
        qid = str(r["query_id"])
        if qid not in contexts:
            logger.warning(f"No context found for query {qid}")
            continue

        context_docs = contexts[qid]
        model_answer = r.get(answer_key, "").strip()
        
        if not model_answer:
            logger.warning(f"Empty model answer for query {qid}")
            continue

        # Build context text
        passages_text_blocks = []
        for i, doc in enumerate(context_docs, start=1):
            title = doc.get("title", "")
            text_content = doc.get("text", "")
            passages_text_blocks.append(f"[Document {i}]\nTitle: {title}\n{text_content}")

        source_docs_text = "\n\n---\n\n".join(passages_text_blocks)

        prompt_text = PROMPT_TEMPLATE.format(
            source_docs_text=source_docs_text,
            model_answer=model_answer
        )

        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)
        valid_queries.append(r)

    if not prompts:
        logger.info("No valid prompts constructed for evaluation.")
        sys.exit(0)

    # Note: Guided decoding (outlines) is disabled here because FSM vocabulary masking 
    # for Gemma-2's 256k vocabulary on CPU has extremely high overhead, and can force 
    # the model to output empty strings. Gemma-2-27B is highly capable and follows JSON 
    # formatting instructions natively without constraints.
    params = SamplingParams(
        temperature=0.0,
        max_tokens=2000,
    )

    logger.info(f"Running batch generation for {len(prompts)} prompts...")
    t_start = time.time()
    outputs = llm.generate(prompts, params)
    logger.info(f"Generation completed in {time.time() - t_start:.2f}s")

    # Parse and record results
    total_fully = 0
    total_partially = 0
    total_unsupported = 0
    failed = 0

    # Include existing metrics if resuming
    for qid, res in results.items():
        total_fully += res.get("fully_supported", 0)
        total_partially += res.get("partially_supported", 0)
        total_unsupported += res.get("unsupported", 0)

    for r, out in zip(valid_queries, outputs):
        query_id = str(r["query_id"])
        resp_text = out.outputs[0].text.strip()

        try:
            # Clean markdown code block wraps if present
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", resp_text)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()
            
            eval_json = json.loads(cleaned_text)
            statements = eval_json.get("statements", [])
            
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
            
            total_fully += fully
            total_partially += partially
            total_unsupported += unsupported
        except Exception as e:
            logger.error(f"Failed to parse model output for query {query_id}: {e}")
            logger.debug(f"Raw output: {resp_text}")
            failed += 1

    save_atomic(results, output_path)

    total_statements_all = total_fully + total_partially + total_unsupported
    global_support_score = ((total_fully + 0.5 * total_partially) / total_statements_all * 100) if total_statements_all > 0 else 0
    hallucination_rate = (total_unsupported / total_statements_all * 100) if total_statements_all > 0 else 0

    print("\n" + "="*50)
    print("LOCAL BATCH SUPPORT EVALUATION COMPLETE")
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
