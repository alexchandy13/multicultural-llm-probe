"""Reformat CulturalBench questions into binary True/False prompts.

For each (question, option) row, produces:
  "In [country], is [option] [predicate]?"

Uses claude-haiku-4-5 to extract the predicate for each unique question.
Output: data/culturalbench_reformatted.json

Usage:
    python data/prep_culturalbench.py
    python data/prep_culturalbench.py --dry-run   # show 20 examples, no API calls
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "culturalbench_reformatted.json"

# Countries that take "the" as a determiner
THE_COUNTRIES = {
    "Netherlands", "United Kingdom", "United States",
    "United Arab Emirates", "Dominican Republic", "Philippines",
    "Czech Republic",
}

SYSTEM = """\
You reformat cultural knowledge questions into a binary predicate fragment.

Given a question like "In the Netherlands, which of the following is an unusual common public practice?"
extract the predicate so the question can be rewritten as:
  "In [country], is [option] [predicate]?"

Rules:
- Output ONLY the predicate fragment — nothing else, no punctuation at the end
- The fragment must grammatically follow "is [option]"
- "which of the following is X?" → "X"
- "what is X?" → "X"
- "what are X?" → "X"
- "how do people X?" → "a common way to X"
- "when do people X?" → "a common time to X"
- "who is typically X?" → "typically X"
- Questions without explicit country prefix (e.g. "What do Indians traditionally prefer for...") are still
  about the country given — extract the predicate the same way
- Do NOT include a question mark or trailing period
- Keep it concise and preserve meaning"""


def country_with_article(country: str) -> str:
    return f"the {country}" if country in THE_COUNTRIES else country


def extract_predicates(client: anthropic.Anthropic,
                       questions: dict[int, dict]) -> dict[int, str]:
    predicates: dict[int, str] = {}
    items = list(questions.items())
    for i, (qidx, info) in enumerate(items):
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            system=SYSTEM,
            messages=[{"role": "user", "content": info["question"]}],
        )
        predicates[qidx] = msg.content[0].text.strip().rstrip(".")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)}")
        time.sleep(0.05)
    return predicates


def build_reformatted_prompt(country: str, option: str, predicate: str) -> str:
    phrase = country_with_article(country)
    opt = option.strip().rstrip(".")
    opt = opt[0].lower() + opt[1:] if opt else opt
    return f"In {phrase}, is {opt} {predicate}?"


def dry_run(ds) -> None:
    for i in range(20):
        r = ds[i]
        print(f"country:   {r['country']}")
        print(f"question:  {r['prompt_question']}")
        print(f"option:    {r['prompt_option']}")
        print(f"answer:    {r['answer']}")
        # Show what the assembled prompt would look like with a placeholder predicate
        phrase = country_with_article(r["country"])
        opt = r["prompt_option"]
        opt = opt[0].lower() + opt[1:]
        print(f"→ draft:   In {phrase}, is {opt} [PREDICATE]?")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print 20 example rows without calling the API")
    args = parser.parse_args()

    print("Loading CulturalBench...")
    ds = load_dataset("kellycyy/CulturalBench", split="test")
    print(f"  {len(ds)} rows")

    if args.dry_run:
        dry_run(ds)
        return

    # Collect unique questions
    questions: dict[int, dict] = {}
    for r in ds:
        qidx = r["question_idx"]
        if qidx not in questions:
            questions[qidx] = {"question": r["prompt_question"], "country": r["country"]}
    print(f"  {len(questions)} unique questions — calling API...")

    client = anthropic.Anthropic()
    predicates = extract_predicates(client, questions)

    # Assemble output rows
    rows = []
    for r in ds:
        qidx = r["question_idx"]
        predicate = predicates[qidx]
        reformatted = build_reformatted_prompt(r["country"], r["prompt_option"], predicate)
        rows.append({
            "data_idx":          r["data_idx"],
            "question_idx":      qidx,
            "country":           r["country"],
            "original_question": r["prompt_question"],
            "prompt_option":     r["prompt_option"],
            "answer":            r["answer"],
            "predicate":         predicate,
            "reformatted_prompt": reformatted,
        })

    OUTPUT.write_text(json.dumps(rows, indent=2))
    print(f"Saved {len(rows)} rows → {OUTPUT}")

    # Print 10 examples
    print("\nSample reformatted prompts:")
    for row in rows[:10]:
        mark = "T" if row["answer"] else "F"
        print(f"  [{mark}] {row['reformatted_prompt']}")


if __name__ == "__main__":
    main()
