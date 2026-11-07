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

from evaluate.country_names import (
    find_country_reference, strip_country_reference, strip_unregistered_reference,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=12)
    args = ap.parse_args()

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")

    still_excluded = []
    for r in ds:
        q, c = r["prompt_question"], r["country"]
        _, applied = strip_country_reference(q, c)
        if not applied:
            _, applied = strip_unregistered_reference(q)
        if not applied:
            still_excluded.append((c, q))

    n = len(ds)
    deletable = n - len(still_excluded)
    print(f"deletable (--no-country, after both passes): {deletable}/{n} = {deletable / n:.3f}")
    print(f"\nSTILL EXCLUDED -- benchmark typos, city/region names, unlisted demonyms: "
          f"{len(still_excluded)}")
    for c, q in still_excluded[: args.show]:
        print(f"  [{c}] {q[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
