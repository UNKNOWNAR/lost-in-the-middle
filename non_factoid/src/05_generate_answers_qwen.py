# ── Qwen2.5-7B Lost-in-the-Middle Generation (Kaggle T4 x2) ──────────────
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

import sys, json, time, logging
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("qwen_gen")

# ── Paths ─────────────────────────────────────────────────────────────────────
IS_KAGGLE  = os.path.exists("/kaggle/input")
MODEL_NAME = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1" 
DATA_BASE  = Path("/kaggle/input/datasets/arinjaysarkar/k40data") # Updated to k40 dataset
OUTPUT_DIR = Path("/kaggle/working/generated_answers/qwen") 

# ── Validate paths ────────────────────────────────────────────────────────────
if IS_KAGGLE:
    for lbl, p in [("Model", Path(MODEL_NAME)), ("Dataset", DATA_BASE)]:
        if not p.exists():
            logger.error(f"{lbl} NOT FOUND: {p}")
            sys.exit(1)

# ── THE KEY FIX ───────────────────────────────────────────────────────────────
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

# ── Input discovery ───────────────────────────────────────────────────────────
def get_input_dir() -> Path:
    search = DATA_BASE if IS_KAGGLE else DATA_BASE / "contexts" / "k40"

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

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tp,
        max_model_len=32768,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1000, top_p=1.0)
    tokenizer = llm.get_tokenizer()

    total = len(input_files)
    done  = sum(1 for fp in input_files if (OUTPUT_DIR / f"k40_{fp.stem.split('_')[-1]}.jsonl").exists())
    start_t = time.time()

    for fp in input_files:
        c_idx = fp.stem.split("_")[-1]
        out_f = OUTPUT_DIR / f"k40_{c_idx}.jsonl"
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
                    "k": 40,
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
