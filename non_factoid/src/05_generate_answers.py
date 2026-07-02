"""
05_generate_answers.py

Generates answers for all 10 context conditions (119 queries x 10 conditions = 1,190 runs)
using Meta-Llama-3.1-8B-Instruct.
Designed to run on Kaggle with dual T4 GPUs using vLLM, with a fallback to HuggingFace pipeline.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("05_generate_answers")

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"
OUTPUT_DIR = PIPELINE_ROOT / "data" / "outputs" / "generated_answers" / "llama"

# Model name
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def format_rendered_context(context_docs: list) -> str:
    """Formats the list of document dicts into a prompt-ready string."""
    blocks = []
    for doc in context_docs:
        pos = doc["position"]
        title = doc["title"]
        text = doc["text"]
        blocks.append(f"[Document {pos}]\nTitle: {title}\n{text}")
    return "\n\n---\n\n".join(blocks)

def run_vllm(input_files: list[Path]):
    """Runs high-throughput batch generation using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        logger.error("vLLM not installed. Cannot use run_vllm.")
        return False

    # Check GPU availability for TP size
    import torch
    num_gpus = torch.cuda.device_count()
    tensor_parallel = min(num_gpus, 2) if num_gpus > 0 else 1
    logger.info(f"Initializing vLLM with {MODEL_NAME} (GPUs: {num_gpus}, TP size: {tensor_parallel})...")

    # Initialize vLLM
    # Limit max model len to avoid OOM on T4 GPUs (T4 has 16GB VRAM each)
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tensor_parallel,
        max_model_len=16384,
        trust_remote_code=True,
        dtype="float16"              # Add this to avoid bfloat16/T4 issues
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1000,  # increased to prevent cutoff for 400 word generation
        top_p=1.0
    )

    tokenizer = llm.get_tokenizer()

    for file_path in input_files:
        c_idx = file_path.stem.split("_")[-1]
        out_file = OUTPUT_DIR / f"k20_{c_idx}.jsonl"

        if out_file.exists():
            logger.info(f"Output file {out_file} already exists. Skipping condition {c_idx}.")
            continue

        logger.info(f"Processing condition {c_idx} ({file_path.name}) using vLLM...")

        # Load inputs
        queries = []
        prompts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                qid = data["query_id"]
                qtext = data["query_text"]
                context_docs = data["context_docs"]
                
                rendered_context = format_rendered_context(context_docs)
                
                # Format Llama 3.1 Chat Template
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Read all the following documents carefully "
                            "and write a comprehensive answer to the question below.\n\n"
                            "Important instructions:\n"
                            "- Your answer must be approximately 400 words\n"
                            "- Synthesize information across multiple documents\n"
                            "- Every claim you make must be grounded in the provided documents\n"
                            "- Do not use any knowledge outside the provided documents\n"
                            "- Cite which document number(s) support each key claim, e.g. [Doc 3]"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"QUESTION: {qtext}\n\nDOCUMENTS:\n{rendered_context}\n\nWrite your comprehensive answer now:"
                    }
                ]
                
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                queries.append((qid, qtext))
                prompts.append(formatted_prompt)

        # Batch generate
        logger.info(f"Generating answers for {len(prompts)} queries in condition {c_idx}...")
        outputs = llm.generate(prompts, sampling_params)

        # Write outputs
        with open(out_file, "w", encoding="utf-8") as out_f:
            for (qid, qtext), output in zip(queries, outputs):
                answer_text = output.outputs[0].text.strip()
                out_f.write(json.dumps({
                    "query_id": qid,
                    "query_text": qtext,
                    "llama_answer": answer_text
                }, ensure_ascii=False) + "\n")
        
        logger.info(f"Completed condition {c_idx}. Results saved to {out_file}.")

    return True

def run_hf_pipeline(input_files: list[Path]):
    """Fallback runner using standard HuggingFace pipeline."""
    import torch
    from transformers import pipeline, BitsAndBytesConfig

    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"Initializing HF Pipeline (Device: {device})...")

    # Load generation pipeline in 4-bit to prevent memory overflow
    quantization_config = None
    if device == 0:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto" if device == 0 else None,
        model_kwargs={"quantization_config": quantization_config} if quantization_config is not None else {}
    )

    for file_path in input_files:
        c_idx = file_path.stem.split("_")[-1]
        out_file = OUTPUT_DIR / f"k20_{c_idx}.jsonl"

        if out_file.exists():
            logger.info(f"Output file {out_file} already exists. Skipping condition {c_idx}.")
            continue

        logger.info(f"Processing condition {c_idx} ({file_path.name}) using HuggingFace...")

        with open(file_path, "r", encoding="utf-8") as f, open(out_file, "w", encoding="utf-8") as out_f:
            for line in f:
                data = json.loads(line)
                qid = data["query_id"]
                qtext = data["query_text"]
                context_docs = data["context_docs"]
                
                rendered_context = format_rendered_context(context_docs)
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a research assistant. Read all the following documents carefully "
                            "and write a comprehensive answer to the question below.\n\n"
                            "Important instructions:\n"
                            "- Your answer must be approximately 400 words\n"
                            "- Synthesize information across multiple documents\n"
                            "- Every claim you make must be grounded in the provided documents\n"
                            "- Do not use any knowledge outside the provided documents\n"
                            "- Cite which document number(s) support each key claim, e.g. [Doc 3]"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"QUESTION: {qtext}\n\nDOCUMENTS:\n{rendered_context}\n\nWrite your comprehensive answer now:"
                    }
                ]
                
                outputs = generator(messages, max_new_tokens=600, temperature=0.0, do_sample=False)
                answer_text = outputs[0]["generated_text"][-1]["content"].strip()

                out_f.write(json.dumps({
                    "query_id": qid,
                    "query_text": qtext,
                    "llama_answer": answer_text
                }, ensure_ascii=False) + "\n")

        logger.info(f"Completed condition {c_idx}. Results saved to {out_file}.")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = DATA_PROCESSED / "contexts" / "k20"
    
    input_files = sorted(list(input_dir.glob("condition_*.jsonl")), key=lambda x: int(x.stem.split("_")[-1]))
    if not input_files:
        logger.error(f"No condition files found in {input_dir}. Run 04_context_builder.py first.")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} condition files to process.")

    # Try running vLLM first (ideal for Kaggle dual GPUs)
    vllm_success = False
    try:
        vllm_success = run_vllm(input_files)
    except Exception as e:
        logger.warning(f"Failed to run vLLM: {e}. Falling back to HuggingFace pipeline.", exc_info=True)

    # Fallback to HuggingFace pipeline if vLLM isn't present or failed
    if not vllm_success:
        run_hf_pipeline(input_files)

    logger.info("Answer generation completed for all conditions!")

if __name__ == "__main__":
    main()
