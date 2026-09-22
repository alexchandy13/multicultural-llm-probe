"""Convert CulturalBench into NormAd-style narrative stories.

For each (question, option, answer, country) row, uses claude-haiku to write a
short third-person story with an Anglo character name, ending with
"Is what [name] did socially acceptable?"  Gold label is yes/no based on
whether the behavior is culturally appropriate in that country.

Both True (correct) and False (incorrect) options are included, giving a
balanced yes/no split similar to NormAd.

Output: data/culturalbench_normad.json

Usage:
    python data/prep_culturalbench_normad.py
    python data/prep_culturalbench_normad.py --dry-run
    python data/prep_culturalbench_normad.py --limit 50   # small test run
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
You convert cultural knowledge questions into social norms stories in the style of NormAd.

Given a cultural question about a country, one answer option, and whether that option \
is the CORRECT answer (True) or INCORRECT (False):

1. Write a 2-3 sentence third-person narrative story about a character with a common \
Anglo/English name (e.g. Sarah, Tom, Emma, Michael, Chris, Rachel, Jake, Emily, Mark, \
Alice, David, Laura, Ben, Kate, James) who is in or visiting the country and performs \
the behavior described in the option.
2. End the story with exactly: "Is what [name] did socially acceptable?"
3. Output the correct gold label — "yes" if the behavior is culturally normal or \
appropriate in that country, "no" if it is not.

Important: answer=True means this option IS the correct answer to the question. \
Use the question wording to determine whether "correct" maps to acceptable \
(e.g. "common practice", "typical greeting") or unacceptable \
(e.g. "unusual", "rude", "never done", "considered offensive").

Output JSON only, no extra text:
{"story": "...", "gold": "yes" or "no", "name": "..."}"""


def convert_row(client: anthropic.Anthropic, country: str,
                question: str, option: str, answer: bool) -> dict:
    user_msg = (
        f"Country: {country}\n"
        f"Question: {question}\n"
        f"Option: {option}\n"
        f"This option is the CORRECT answer: {answer}"
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
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
            print(f"[{'T' if r['answer'] else 'F'}] {r['country']} | {r['prompt_question'][:60]} | {r['prompt_option']}")
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
                    r["answer"],
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
            "story":        result.get("story"),
            "name":         result.get("name"),
            "gold":         result.get("gold"),
            "original_question": r["prompt_question"],
            "original_option":   r["prompt_option"],
            "original_answer":   r["answer"],
        })

    CACHE.write_text(json.dumps(cache))
    print(f"\nDone. {len(output_rows)} rows converted, {failed} failed.")

    yes_ct = sum(1 for r in output_rows if r.get("gold") == "yes")
    no_ct  = sum(1 for r in output_rows if r.get("gold") == "no")
    print(f"Gold distribution: yes={yes_ct}, no={no_ct}")

    OUTPUT.write_text(json.dumps(output_rows, indent=2))
    print(f"Saved → {OUTPUT}")

    print("\nSample stories:")
    for row in output_rows[:6]:
        print(f"  [{row['gold']}] [{row['country']}] {row['story']}")
        print()


if __name__ == "__main__":
    main()
