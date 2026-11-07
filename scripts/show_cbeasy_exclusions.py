#!/usr/bin/env python3
"""Print sample rows excluded from CulturalBench-Easy's --no-country arm.

Two different reasons a row stays excluded:
  PLURAL: the demonym is the sentence's subject or object ('What do Indians
          prefer', 'the communication style of the Dutch') -- deleting it
          leaves no noun phrase, so it is not grammatically deletable.
  NOT FOUND: the question never names the country at all -- it refers to an
          ethnic group, a city/region, or a religion instead ('young Yoruba
          man', 'In Zulu tradition', 'In Judaism', 'In Madrid').

  python scripts/show_cbeasy_exclusions.py
  python scripts/show_cbeasy_exclusions.py --show 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from evaluate.country_names import find_country_reference, strip_country_reference


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")

    plural, not_found = [], []
    for r in ds:
        q, c = r["prompt_question"], r["country"]
        _, applied = strip_country_reference(q, c)
        if applied:
            continue
        hit = find_country_reference(q, c)
        (plural if hit and hit[0] == "plural" else not_found).append((c, q))

    print(f"PLURAL -- demonym is the subject/object, deletion breaks the sentence: {len(plural)}")
    for c, q in plural[: args.show]:
        print(f"  [{c}] {q}")

    print(f"\nNOT FOUND -- no country reference at all (ethnic group / city / religion): {len(not_found)}")
    for c, q in not_found[: args.show]:
        print(f"  [{c}] {q[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
