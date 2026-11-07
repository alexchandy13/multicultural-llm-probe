#!/usr/bin/env python3
"""Print full CulturalBench-Easy rows matching a text substring.

  python scripts/show_cbeasy_row.py "feel safe walking home"
  python scripts/show_cbeasy_row.py "communication style of the Dutch"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: show_cbeasy_row.py <substring>", file=sys.stderr)
        return 2
    needle = sys.argv[1].lower()
    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")
    hits = 0
    for r in ds:
        if needle in r["prompt_question"].lower():
            hits += 1
            print(f"\ndata_idx={r['data_idx']}  question_idx={r['question_idx']}  "
                  f"country={r['country']}  gold={r['answer']}")
            print(f"  Q: {r['prompt_question']}")
            for k in "abcd":
                mark = " <-- gold" if k.upper() == r["answer"] else ""
                print(f"  {k.upper()}. {r[f'prompt_option_{k}']}{mark}")
    if not hits:
        print("no rows matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
