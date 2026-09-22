"""Convert CulturalBench into NormAd-style yes/no prompts with Anglo names.

For each (question, option, answer, country) row:
  - Rewrites the option as a short narrative sentence with an Anglo name
  - Appends a yes/no question derived from the original question text
  - Gold = "yes" if original answer=True, "no" if answer=False

Example:
  question: "In the Netherlands, which of the following is an unusual common public practice?"
  option:   "Cycle everywhere"
  answer:   False
  →  "Sarah cycles everywhere. Is what she did an unusual common public practice?"
  gold: "no"

Output: data/culturalbench_normad.json

Usage:
    python data/prep_culturalbench_normad.py --dry-run
    python data/prep_culturalbench_normad.py --limit 20
    python data/prep_culturalbench_normad.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "culturalbench_normad.json"
CACHE  = PROJECT_ROOT / "data" / "culturalbench_normad_cache.json"

SYSTEM = """\
You reformat cultural knowledge questions into short narrative yes/no prompts.

Given:
- country: the country the question is about
- question: a multiple-choice question about cultural practices in that country
- option: one answer option from that question

Your task:
1. Write a short 1-2 sentence story where a character named Sarah (or another common \
Anglo name like Tom, Emma, Michael, Chris, Rachel, Jake, Emily, Mark, Alice, David) \
performs the behavior described in the option. Use "she" or "he" as appropriate.
2. Append a yes/no question by rephrasing the original question as \
"Is what she/he did [rest of question]?" — strip the country prefix \
(e.g. "In the Netherlands,") and any "which of the following" phrasing, \
keeping the core question intact.

Output JSON only, no extra text:
{"prompt": "...", "name": "..."}

Examples:
question: "In the Netherlands, which of the following is an unusual common public practice?"
option: "Cycle everywhere"
→ {"prompt": "Sarah cycles everywhere. Is what she did an unusual common public practice?", "name": "Sarah"}

question: "In Korean dining etiquette, what is a common practice regarding drinks and paying for the meal?"
option: "Younger people pour drinks for the elders and the elders pay for the meal."
→ {"prompt": "Sarah, who is younger, poured drinks for her elders and the elders paid for the meal. Is what she did a common practice regarding drinks and paying for the meal?", "name": "Sarah"}"""


def convert_row(client: anthropic.Anthropic, country: str,
                question: str, option: str) -> dict:
    user_msg = (
        f"country: {country}\n"
        f"question: {question}\n"
        f"option: {option}"
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = msg.content[0].text.strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    return json.loads(text[start:end])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print raw rows without calling the API")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many rows (for testing)")
    args = parser.parse_args()

    print("Loading CulturalBench...")
    ds = load_dataset("kellycyy/CulturalBench", split="test")
    rows = list(ds)
    if args.limit:
        rows = rows[:args.limit]
    print(f"  {len(rows)} rows to process")

    if args.dry_run:
        for r in rows[:8]:
            gold = "yes" if r["answer"] else "no"
            print(f"[{gold}] {r['country']} | {r['prompt_question'][:70]} | {r['prompt_option']}")
        return

    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
        print(f"  Loaded {len(cache)} cached conversions")

    client = anthropic.Anthropic()
    output_rows = []
    failed = 0

    for i, r in enumerate(rows):
        key = str(r["data_idx"])
        if key not in cache:
            try:
                result = convert_row(
                    client,
                    r["country"],
                    r["prompt_question"],
                    r["prompt_option"] or "",
                )
                cache[key] = result
            except Exception as e:
                print(f"  Warning: data_idx={r['data_idx']} failed ({e}), skipping")
                failed += 1
                cache[key] = None

            if (i + 1) % 200 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  {i + 1}/{len(rows)}  (failed so far: {failed})")
            time.sleep(0.05)

        result = cache.get(key)
        if result is None:
            continue

        output_rows.append({
            "data_idx":     r["data_idx"],
            "question_idx": r["question_idx"],
            "country":      r["country"],
            "prompt":       result.get("prompt"),
            "name":         result.get("name"),
            "gold":         "yes" if r["answer"] else "no",
            "original_question": r["prompt_question"],
            "original_option":   r["prompt_option"],
        })

    CACHE.write_text(json.dumps(cache))
    print(f"\nDone. {len(output_rows)} rows converted, {failed} failed.")

    yes_ct = sum(1 for r in output_rows if r["gold"] == "yes")
    no_ct  = sum(1 for r in output_rows if r["gold"] == "no")
    print(f"Gold distribution: yes={yes_ct}, no={no_ct}")

    OUTPUT.write_text(json.dumps(output_rows, indent=2))
    print(f"Saved → {OUTPUT}")

    print("\nSample prompts:")
    for row in output_rows[:8]:
        print(f"  [{row['gold']}] [{row['country']}] {row['prompt']}")
        print()


if __name__ == "__main__":
    main()
