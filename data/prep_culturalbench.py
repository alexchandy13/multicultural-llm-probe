"""Reformat CulturalBench questions into binary Yes/No prompts.

For each (question, option) row, produces a grammatically natural Y/N question
by passing the full question + all options to claude-haiku-4-5.

Output: data/culturalbench_reformatted.json

Usage:
    python data/prep_culturalbench.py
    python data/prep_culturalbench.py --dry-run   # show raw rows, no API calls
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anthropic
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "culturalbench_reformatted.json"
CACHE  = PROJECT_ROOT / "data" / "culturalbench_reformat_cache.json"

# Countries that take "the" as a determiner
THE_COUNTRIES = {
    "Netherlands", "United Kingdom", "United States",
    "United Arab Emirates", "Dominican Republic", "Philippines",
    "Czech Republic",
}

SYSTEM = """\
You reformat multiple-choice cultural knowledge questions into natural Yes/No questions.

Given an original question and its answer options, rewrite each (question, option) pair
as a single, grammatically correct Yes/No question that can be answered with "Yes" or "No".

Rules:
- Always start with "In [country]," using the canonical country name provided
- The question must be natural and fluent — fix grammar, tense, and phrasing as needed
- Preserve the meaning of the original question + option combination
- Do NOT include the answer — just ask the question
- Output a JSON array of strings, one per option, in the same order as the input options
- No extra text, just the JSON array"""


def country_with_article(country: str) -> str:
    return f"the {country}" if country in THE_COUNTRIES else country


def reformat_question(client: anthropic.Anthropic, country: str,
                      question: str, options: list[str]) -> list[str]:
    phrase = country_with_article(country)
    user_msg = (
        f"Country: {phrase}\n"
        f"Original question: {question}\n"
        f"Options:\n" + "\n".join(f"- {o}" for o in options)
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = msg.content[0].text.strip()
    # Parse JSON array
    start = text.find("[")
    end   = text.rfind("]") + 1
    return json.loads(text[start:end])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print raw rows without calling the API")
    args = parser.parse_args()

    print("Loading CulturalBench...")
    ds = load_dataset("kellycyy/CulturalBench", split="test")
    print(f"  {len(ds)} rows")

    if args.dry_run:
        for r in ds[:8]:
            print(f"[{'T' if r['answer'] else 'F'}] {r['country']} | {r['prompt_question']} | {r['prompt_option']}")
        return

    # Group rows by question_idx
    groups: dict[int, dict] = {}
    for r in ds:
        qidx = r["question_idx"]
        if qidx not in groups:
            groups[qidx] = {
                "question": r["prompt_question"],
                "country":  r["country"],
                "options":  [],
                "data_idxs": [],
                "answers":  [],
            }
        groups[qidx]["options"].append(r["prompt_option"] or "")
        groups[qidx]["data_idxs"].append(r["data_idx"])
        groups[qidx]["answers"].append(r["answer"])

    print(f"  {len(groups)} unique questions")

    # Load cache
    cache: dict[str, list[str]] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())
        print(f"  Loaded {len(cache)} from cache")

    client = anthropic.Anthropic()
    items = list(groups.items())
    for i, (qidx, info) in enumerate(items):
        key = str(qidx)
        if key in cache:
            continue
        try:
            reformatted = reformat_question(
                client, info["country"], info["question"], info["options"]
            )
            # Pad if LLM returned fewer items than options
            while len(reformatted) < len(info["options"]):
                reformatted.append(None)
            cache[key] = reformatted
        except Exception as e:
            print(f"  Warning: q{qidx} failed ({e}), skipping")
            cache[key] = [None] * len(info["options"])
        if (i + 1) % 100 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  {i + 1}/{len(items)}")
        time.sleep(0.05)

    CACHE.write_text(json.dumps(cache))
    print(f"  Done. Saved cache ({len(cache)} entries)")

    # Assemble flat output rows
    # Build a lookup: data_idx → reformatted prompt
    reformatted_by_dataidx: dict[int, str] = {}
    for qidx, info in groups.items():
        key = str(qidx)
        rephrased = cache.get(key, [])
        for j, didx in enumerate(info["data_idxs"]):
            reformatted_by_dataidx[didx] = rephrased[j] if j < len(rephrased) else None

    rows = []
    for r in ds:
        rows.append({
            "data_idx":           r["data_idx"],
            "question_idx":       r["question_idx"],
            "country":            r["country"],
            "original_question":  r["prompt_question"],
            "prompt_option":      r["prompt_option"],
            "answer":             r["answer"],
            "reformatted_prompt": reformatted_by_dataidx.get(r["data_idx"]),
        })

    OUTPUT.write_text(json.dumps(rows, indent=2))
    print(f"Saved {len(rows)} rows → {OUTPUT}")

    print("\nSample reformatted prompts:")
    for row in rows[:50]:
        mark = "T" if row["answer"] else "F"
        print(f"  [{mark}] {row['reformatted_prompt']}")


if __name__ == "__main__":
    main()
