# ── Qwen2.5-7B Lost-in-the-Middle Generation (Kaggle T4 x2) ──────────────
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import sys, json, time, logging
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("qwen_gen_k100")

# ── Paths ─────────────────────────────────────────────────────────────────────
IS_KAGGLE  = os.path.exists("/kaggle/input")
MODEL_NAME = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1" 
DATA_BASE  = Path("/kaggle/input/datasets/unknownarunkownar/k-100-data/k100") # Updated to k100 dataset
OUTPUT_DIR = Path("/kaggle/working/generated_answers/qwen") 

# ── Validate paths ────────────────────────────────────────────────────────────
if IS_KAGGLE:
    for lbl, p in [("Model", Path(MODEL_NAME)), ("Dataset", DATA_BASE)]:
        if not p.exists():
            logger.error(f"{lbl} NOT FOUND: {p}")
            sys.exit(1)

# Note: VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (set above) handles the max_model_len
# override natively — no monkey-patching needed for Qwen.

# ── Input discovery ───────────────────────────────────────────────────────────
def get_input_dir() -> Path:
    search = DATA_BASE if IS_KAGGLE else DATA_BASE / "contexts" / "k100"

    files = list(search.glob("condition_*.jsonl"))

    if files:
        return search

    raise FileNotFoundError(
        f"No condition_*.jsonl files found under {search}"
    )

# ── Prompting helpers ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a research assistant. Read all the following documents carefully "
    "and write a comprehensive answer to the question below.\n\n"
    "Important instructions:\n"
    "- Your answer must be approximately 400 words\n"
    "- Synthesize information across multiple documents\n"
    "- Every claim you make must be grounded in the provided documents\n"
    "- Do not use any knowledge outside the provided documents\n"
    "- Cite which document number(s) support each key claim, e.g. [Doc 3]"
)

def format_context(docs):
    return "\n\n---\n\n".join(
        f"[Document {d['position']}]\nTitle: {d['title']}\n{d['text']}" for d in docs)

def build_messages(query, docs):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"QUESTION: {query}\n\nDOCUMENTS:\n{format_context(docs)}\n\nWrite your comprehensive answer now:"},
    ]

# ── vLLM runner ───────────────────────────────────────────────────────────────
def run_vllm(input_files):
    from vllm import LLM, SamplingParams
    import torch

    ngpu = torch.cuda.device_count()
    tp = min(ngpu, 2) if ngpu > 0 else 1

    # Absolute max tokens found in k=100 dataset is 87,864 tokens.
    # Setting max_model_len=90112 (allowing ~2,200 token buffer for generation).
    # Removing max_num_seqs=1 to allow vLLM to batch dynamically for speed.
    # gpu_memory_utilization=0.95 squeezes more KV cache out of the T4 GPUs.
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tp,
        max_model_len=90112,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.95,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1000, top_p=1.0)
    tokenizer = llm.get_tokenizer()

    # STRICT safety cap: Skip queries > 70,000 tokens to NEVER hit a hard CUDA crash
    MAX_PROMPT_TOKENS = 70000

    total = len(input_files)
    done  = sum(1 for fp in input_files if (OUTPUT_DIR / f"k100_{fp.stem.split('_')[-1]}.jsonl").exists())

    for fp in input_files:
        c_idx = fp.stem.split("_")[-1]
        out_f = OUTPUT_DIR / f"k100_{c_idx}.jsonl"
        if out_f.exists(): continue

        t0 = time.time()
        queries = []
        prompts = []
        skipped_qids = set()
        
        # 1. Pre-process and drop massive prompts
        rows = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        for d in rows:
            docs = d["context_docs"]
            query = d["query_text"]
            qid   = d["query_id"]

            msgs = build_messages(query, docs)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            n_tokens = len(tokenizer.encode(prompt))

            if n_tokens > MAX_PROMPT_TOKENS:
                logger.warning(f"Query {qid}: {n_tokens} tokens is too large for T4 GPU! Dropping query.")
                skipped_qids.add(qid)
                # Give vLLM a tiny dummy prompt so it doesn't break batch alignment
                prompt = tokenizer.apply_chat_template([{"role":"user", "content":"skip"}], tokenize=False, add_generation_prompt=True)

            queries.append((qid, query))
            prompts.append(prompt)

        # 2. Generate all at once (vLLM will batch automatically)
        logger.info(f"Condition {c_idx}: Starting batched generation for {len(prompts)} queries...")
        outputs = llm.generate(prompts, params)

        # 3. Save results
        tmp_f = out_f.with_suffix(".tmp")
        with tmp_f.open("w", encoding="utf-8") as fh:
            for (qid, qtxt), out in zip(queries, outputs):
                ans = "" if qid in skipped_qids else out.outputs[0].text.strip()
                fh.write(json.dumps({
                    "query_id": qid,
                    "query_text": qtxt,
                    "k": 100,
                    "condition": int(c_idx),
                    "qwen_answer": ans,
                }, ensure_ascii=False) + "\n")
        tmp_f.rename(out_f)

        done += 1
        logger.info(f"Condition {c_idx} done in {time.time()-t0:.1f}s. Overall: {done}/{total}")


# ── Main ──────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION CONTROL — change this before running each session  ║
# ║  Session 1: CONDITIONS_TO_RUN = [1, 2, 3, 4, 5]            ║
# ║  Session 2: CONDITIONS_TO_RUN = [6, 7, 8, 9, 10]           ║
# ╚══════════════════════════════════════════════════════════════╝
CONDITIONS_TO_RUN = [1, 2, 3, 4, 5]  # ← Edit this for each session

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
all_files = sorted(get_input_dir().glob("condition_*.jsonl"), key=lambda x: int(x.stem.split("_")[-1]))
files = [f for f in all_files if int(f.stem.split("_")[-1]) in CONDITIONS_TO_RUN]
logger.info(f"Running conditions: {CONDITIONS_TO_RUN}")
run_vllm(files)
logger.info("All conditions complete!")

