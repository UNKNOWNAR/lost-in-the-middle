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
    # NOTE: On Kaggle T4 GPUs (16GB VRAM), 90k tokens of KV cache is extremely large.
    # If you experience OOM errors, add `quantization="bitsandbytes"` to the LLM config below.
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tp,
        max_model_len=90112,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1000, top_p=1.0)
    tokenizer = llm.get_tokenizer()

    total = len(input_files)
    done  = sum(1 for fp in input_files if (OUTPUT_DIR / f"k100_{fp.stem.split('_')[-1]}.jsonl").exists())
    start_t = time.time()

    for fp in input_files:
        c_idx = fp.stem.split("_")[-1]
        out_f = OUTPUT_DIR / f"k100_{c_idx}.jsonl"
        if out_f.exists(): continue

        t0 = time.time()
        queries, prompts = [], []
        for raw in fp.read_text(encoding="utf-8").splitlines():
            d = json.loads(raw)
            msgs = build_messages(d["query_text"], d["context_docs"])
            queries.append((d["query_id"], d["query_text"]))
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        outputs = llm.generate(prompts, params)

        tmp_f = out_f.with_suffix(".tmp")
        with tmp_f.open("w", encoding="utf-8") as fh:
            for (qid, qtxt), out in zip(queries, outputs):
                fh.write(json.dumps({
                    "query_id": qid,
                    "query_text": qtxt,
                    "k": 100,
                    "condition": int(c_idx),
                    "qwen_answer": out.outputs[0].text.strip(),
                }, ensure_ascii=False) + "\n")
        tmp_f.rename(out_f)

        done += 1
        logger.info(f"Condition {c_idx} done in {time.time()-t0:.1f}s. Overall: {done}/{total}")

# ── Main ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
files = sorted(get_input_dir().glob("condition_*.jsonl"), key=lambda x: int(x.stem.split("_")[-1]))
run_vllm(files)
logger.info("All conditions complete!")
