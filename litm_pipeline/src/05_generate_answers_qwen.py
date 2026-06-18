"""
05_generate_answers_qwen.py

Generates answers for all 10 context conditions (119 queries x 10 conditions = 1,190 runs)
using Qwen/Qwen2.5-7B-Instruct.
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
logger = logging.getLogger("05_generate_answers_qwen")

# ============================================================
# Kaggle / T4 GPU Compatibility Fixes (must be set BEFORE vLLM import)
# FlashInfer's JIT-compiled kernels fail on T4 (compute 7.5) because it
# cannot link -lcuda at runtime. We disable both the FlashInfer sampler
# AND the FlashInfer attention backend, forcing vLLM to use its
# built-in Triton-based kernels instead, which work fine on T4.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
# ============================================================

# Kaggle vs Local Paths
if os.path.exists("/kaggle/input"):
    # Kaggle environment
    logger.info("Running in Kaggle environment.")
    DATA_PROCESSED = Path("/kaggle/input/datasets/arinjaysarkar/litm-dataset")
    OUTPUT_DIR = Path("/kaggle/working/generated_answers/qwen")
else:
    # Local environment
    SCRIPT_DIR = Path(__file__).resolve().parent
    PIPELINE_ROOT = SCRIPT_DIR.parent
    DATA_PROCESSED = PIPELINE_ROOT / "data" / "processed"
    OUTPUT_DIR = PIPELINE_ROOT / "data" / "outputs" / "generated_answers" / "qwen"

# ── THE KEY FIX ───────────────────────────────────────────────────────────────
# Monkey-patch vllm.config._get_and_verify_max_len to tolerate missing 'factor'
import vllm.config as _vllm_cfg
if not hasattr(_vllm_cfg, '_orig_get_max_len'):
    _orig_get_max_len = _vllm_cfg._get_and_verify_max_len
    def _patched_get_max_len(hf_config, max_model_len, *args, **kwargs):
        rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(rs, dict):
            rope_type = rs.get("rope_type") or rs.get("type") or "default"
            if rope_type in ("default", "linear") and "factor" not in rs:
                hf_config = type(hf_config)(**{**hf_config.to_dict(),
                    "rope_scaling": {**rs, "factor": 1.0}})
        return _orig_get_max_len(hf_config, max_model_len, *args, **kwargs)
    _vllm_cfg._get_and_verify_max_len = _patched_get_max_len
# ──────────────────────────────────────────────────────────────────────────────


def get_input_dir():
    """Dynamically finds the directory containing condition files."""
    if os.path.exists("/kaggle/input"):
        # Search for condition_*.jsonl in case of nested zip structure
        for p in DATA_PROCESSED.rglob("condition_0.jsonl"):
            return p.parent
        for p in Path("/kaggle/input").rglob("condition_0.jsonl"):
            return p.parent
        return DATA_PROCESSED
    return DATA_PROCESSED / "contexts" / "k20"

# Model name (Qwen 2.5 7B Instruct - representing the 8B class)
# Check if mounted locally on Kaggle, otherwise use HF Hub
KAGGLE_MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
if os.path.exists(KAGGLE_MODEL_PATH):
    MODEL_NAME = KAGGLE_MODEL_PATH
    logger.info(f"Using local Kaggle model path: {MODEL_NAME}")
else:
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def format_rendered_context(context_docs: list) -> str:
    """Formats the list of document dicts into a prompt-ready string."""
    blocks = []
    for doc in context_docs:
        pos = doc["position"]
        title = doc["title"]
        text = doc["text"]
        blocks.append(f"[Document {pos}]\nTitle: {title}\n{text}")
    return "\n\n---\n\n".join(blocks)

def run_vllm(input_files: list[Path]) -> None:
    """Runs high-throughput batch generation using vLLM."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        logger.error("vLLM not installed. Cannot use run_vllm.")
        raise e

    import torch
    import time
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
        dtype="float16",             # Add this to avoid bfloat16/T4 issues
        enforce_eager=True,          # Disables CUDA graph compilation (more stable on T4)
        disable_custom_all_reduce=True  # T4 lacks GPU P2P, so skip the custom all-reduce
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1000,  # increased to prevent cutoff for 400 word generation
        top_p=1.0
    )

    tokenizer = llm.get_tokenizer()

    start_time = time.time()
    total_conditions = len(input_files)
    completed_conditions = sum(1 for fp in input_files if (OUTPUT_DIR / f"k20_{fp.stem.split('_')[-1]}.jsonl").exists())
    logger.info(f"Progress check: {completed_conditions}/{total_conditions} conditions already done.")

    for file_path in input_files:
        c_idx = file_path.stem.split("_")[-1]
        out_file = OUTPUT_DIR / f"k20_{c_idx}.jsonl"

        if out_file.exists():
            continue

        cond_start = time.time()
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
                
                # Format Qwen Chat Template
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
                    "qwen_answer": answer_text
                }, ensure_ascii=False) + "\n")
        
        cond_elapsed = time.time() - cond_start
        completed_conditions += 1
        elapsed_overall = time.time() - start_time
        
        qps = len(prompts) / cond_elapsed if cond_elapsed > 0 else 0
        pct = (completed_conditions / total_conditions) * 100
        
        remaining = total_conditions - completed_conditions
        avg_time_per_cond = elapsed_overall / (completed_conditions if completed_conditions > 0 else 1)
        eta_sec = remaining * avg_time_per_cond
        eta_min = eta_sec / 60
        
        logger.info(f"====== PROGRESS METRICS ======")
        logger.info(f"Condition {c_idx} finished in {cond_elapsed:.1f}s ({qps:.2f} queries/sec)")
        logger.info(f"Overall Progress: {completed_conditions}/{total_conditions} conditions ({pct:.1f}%)")
        logger.info(f"Total Elapsed Time: {elapsed_overall/60:.1f} mins")
        if remaining > 0:
            logger.info(f"Estimated Time Remaining (ETA): {eta_min:.1f} mins")
        logger.info(f"==============================")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = get_input_dir()
    
    input_files = sorted(list(input_dir.glob("condition_*.jsonl")), key=lambda x: int(x.stem.split("_")[-1]))
    if not input_files:
        logger.error(f"No condition files found in {input_dir}. Run 04_context_builder.py first.")
        sys.exit(1)

    logger.info(f"Found {len(input_files)} condition files to process.")

    # Run vLLM. No HF fallback.
    run_vllm(input_files)

    logger.info("Answer generation completed for all conditions!")

if __name__ == "__main__":
    main()

