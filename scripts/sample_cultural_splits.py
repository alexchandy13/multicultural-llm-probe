"""Sample 30 English examples from each of the 4 cultural splits for manual inspection."""
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
    english = [ex for ex in ds if is_english(ex.get("output", "") or ex.get("response", ""))]
    sampled = random.sample(english, min(N, len(english)))
    return sampled


def sample_dpo(name):
    ds = load_from_disk(str(DATA_DIR / name))
    english = [ex for ex in ds if is_english(ex.get("chosen", "")[:500])]
    sampled = random.sample(english, min(N, len(english)))
    return sampled


def print_sft(name, examples):
    print(f"\n{'='*80}")
    print(f"  {name.upper()}  ({len(examples)} sampled)")
    print(f"{'='*80}")
    for i, ex in enumerate(examples, 1):
        instruction = ex.get("instruction") or ex.get("inputs") or ""
        response = ex.get("output") or ex.get("response") or ""
        print(f"\n--- {i} ---")
        print(f"INSTRUCTION: {instruction[:300]}")
        print(f"RESPONSE:    {response[:300]}")


def print_dpo(name, examples):
    print(f"\n{'='*80}")
    print(f"  {name.upper()}  ({len(examples)} sampled)")
    print(f"{'='*80}")
    for i, ex in enumerate(examples, 1):
        prompt = ex.get("prompt", "")
        chosen = ex.get("chosen", "")
        rejected = ex.get("rejected", "")
        print(f"\n--- {i} ---")
        print(f"PROMPT:   {prompt[:300]}")
        print(f"CHOSEN:   {chosen[:200]}")
        print(f"REJECTED: {rejected[:200]}")


if __name__ == "__main__":
    print_sft("aya_cult",   sample_sft("aya_cult"))
    print_sft("aya_nocult", sample_sft("aya_nocult"))
    print_dpo("dpo_cult",   sample_dpo("dpo_cult"))
    print_dpo("dpo_nocult", sample_dpo("dpo_nocult"))
