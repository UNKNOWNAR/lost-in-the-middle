"""
Shared utilities for the entire pipeline.
"""
import os
import re
import json
import time
import string
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ─── paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # litm_pipeline/
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS   = ROOT / "data" / "outputs"
PROMPTS_DIR    = ROOT / "prompts"
RESULTS_DIR    = ROOT / "results"

# ─── config ──────────────────────────────────────────────────────────
load_dotenv(ROOT.parent / ".env")        # reuse existing .env at repo root
load_dotenv(ROOT / ".env", override=True) # or pipeline-local .env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_RPM     = 14  # safety margin on 15 RPM free-tier limit


# ─── normalize_answer (from existing codebase) ──────────────────────
def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization."""
    s = s.lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # fix whitespace
    s = " ".join(s.split())
    return s


# ─── Gemini client with rate limiting ────────────────────────────────
class GeminiClient:
    """
    Wrapper around google-genai with:
      - automatic rate limiting (free tier: 15 RPM)
      - retry with exponential backoff on 429/500
      - JSON parsing with fallback
    """

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL
        self._last_call = 0.0
        self._min_interval = 60.0 / GEMINI_RPM  # seconds between calls

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        expect_json: bool = False,
        max_retries: int = 3,
    ) -> str:
        """Send a prompt to Gemini, return the text response."""
        for attempt in range(max_retries):
            # rate limit
            elapsed = time.time() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        response_mime_type="application/json" if expect_json else None,
                    ),
                )
                self._last_call = time.time()
                text = response.text.strip()
                return text

            except Exception as e:
                wait = 2 ** (attempt + 1)
                logging.warning(f"Gemini API error (attempt {attempt+1}): {e}. "
                                f"Retrying in {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"Gemini API failed after {max_retries} retries")

    def generate_json(self, prompt: str, **kwargs) -> dict | list:
        """Send a prompt, parse the response as JSON."""
        text = self.generate(prompt, expect_json=True, **kwargs)
        # strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)


# ─── checkpointing ──────────────────────────────────────────────────
def save_checkpoint(data, filepath: Path):
    """Save data as JSON with atomic write."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(filepath)


def load_checkpoint(filepath: Path):
    """Load checkpoint if it exists, else return None."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ─── logging ─────────────────────────────────────────────────────────
def setup_logging(name: str) -> logging.Logger:
    """Consistent logging for all pipeline steps."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


# ─── prompt loading ──────────────────────────────────────────────────
def load_prompt(name: str) -> str:
    """Load a prompt template from prompts/ directory."""
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


# ─── citation extraction ────────────────────────────────────────────
def extract_citations(answer_text: str) -> list[int]:
    """Extract [Doc N] citation positions from answer text."""
    matches = re.findall(r"\[Doc\s*(\d+)\]", answer_text, re.IGNORECASE)
    return sorted(set(int(m) for m in matches))
