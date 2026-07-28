"""Sample 30 English examples from each cultural split for manual inspection.

Sections:
  aya_cult / aya_nocult     — SFT, English filtered via langdetect
  uf_cult / uf_nocult       — DPO preference pairs from UltraFeedback (English-only)
  prism_cult / prism_nocult — DPO preference pairs from PRISM, English filtered via langdetect
"""
import random
from pathlib import Path
from datasets import load_from_disk
from langdetect import detect, LangDetectException

DATA_DIR = Path("data")
SEED = 42
N = 30

random.seed(SEED)


def is_english(text):
    try:
        return detect(text[:500]) == "en"
    except LangDetectException:
        return False


def sample_sft(name):
    ds = load_from_disk(str(DATA_DIR / name))
    english = [ex for ex in ds if is_english(ex["text"])]
    return random.sample(english, min(N, len(english)))


def sample_dpo(name, filter_lang=False):
    ds = load_from_disk(str(DATA_DIR / name))
    rows = list(ds)
    if filter_lang:
        rows = [ex for ex in rows if is_english(ex["prompt"])]
    return random.sample(rows, min(N, len(rows)))


def print_sft(name, examples):
    print(f"\n{'='*80}")
    print(f"  {name.upper()}  ({len(examples)} sampled, English only)")
    print(f"{'='*80}")
    for i, ex in enumerate(examples, 1):
        parts = ex["text"].split("\n\n", 1)
        instruction = parts[0] if len(parts) > 0 else ""
        response    = parts[1] if len(parts) > 1 else ""
        print(f"\n--- {i} ---")
        print(f"INSTRUCTION: {instruction[:300]}")
        print(f"RESPONSE:    {response[:300]}")


def print_dpo(name, examples):
    print(f"\n{'='*80}")
    print(f"  {name.upper()}  ({len(examples)} sampled)")
    print(f"{'='*80}")
    for i, ex in enumerate(examples, 1):
        print(f"\n--- {i} ---")
        print(f"PROMPT:   {ex['prompt'][:300]}")
        print(f"CHOSEN:   {ex['chosen'][:200]}")
        print(f"REJECTED: {ex['rejected'][:200]}")


if __name__ == "__main__":
    print_sft("aya_cult",     sample_sft("aya_cult"))
    print_sft("aya_nocult",   sample_sft("aya_nocult"))
    print_dpo("uf_cult",      sample_dpo("uf_cult"))
    print_dpo("uf_nocult",    sample_dpo("uf_nocult"))
    print_dpo("prism_cult",   sample_dpo("prism_cult",   filter_lang=True))
    print_dpo("prism_nocult", sample_dpo("prism_nocult", filter_lang=True))
