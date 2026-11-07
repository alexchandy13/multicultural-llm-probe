#!/usr/bin/env python3
"""Print before/after pairs for every CulturalBench-Easy row the no-country
arm successfully rewrites, so the rewrites can be eyeballed before spending
GPU time on the arm.

  python scripts/show_cbeasy_deletions.py
  python scripts/show_cbeasy_deletions.py --show 200
  python scripts/show_cbeasy_deletions.py --kind plural
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
    ap.add_argument("--show", type=int, default=100)
    ap.add_argument("--kind", default=None,
                    help="adjective | attributive | name | alias | plural | unregistered")
    args = ap.parse_args()

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")

    rows = []
    for r in ds:
        q, c = r["prompt_question"], r["country"]
        hit = find_country_reference(q, c)
        new, applied = strip_country_reference(q, c)
        kind = hit[0] if (hit and applied) else None
        if not applied:
            new, applied = strip_unregistered_reference(q)
            kind = "unregistered" if applied else None
        if applied:
            rows.append((kind, c, q, new))

    if args.kind:
        rows = [r for r in rows if r[0] == args.kind]

    print(f"{len(rows)} rows rewritten"
          + (f" of kind {args.kind!r}" if args.kind else f" of {len(ds)} total"))
    for kind, c, old, new in rows[: args.show]:
        print(f"\n[{kind:12}] [{c}]")
        print(f"  before: {old}")
        print(f"  after:  {new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
