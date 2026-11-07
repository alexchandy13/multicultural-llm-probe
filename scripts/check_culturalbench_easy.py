#!/usr/bin/env python3
"""Audit the country-reference swap on CulturalBench-Easy. No GPU, no model.

CulturalBench-Easy asks in the authors' own words, so 69% of its 1,227
questions never write the country down -- they use a demonym ('In Korean
dining etiquette'), an acronym ('In UK culture') or nothing at all. A probe
that replaces only the literal name reaches 28% of the benchmark and scores
the other 72% as if they had been swapped.

This prints, for every country in the split, which surface form was detected
and what each of the eight probe targets would be substituted in as, so the
sixteen target strings can be checked by eye before any GPU time is spent.

  python scripts/check_culturalbench_easy.py
  python scripts/check_culturalbench_easy.py --show 5 --kind plural
  python scripts/check_culturalbench_easy.py --targets us,netherlands

Exits non-zero when a country is referred to by a demonym it has no table
entry for, since that is a row the probe cannot swap correctly.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from evaluate.country_names import (
    DEMONYMS, find_country_reference, resolve_probe_country,
    strip_country_reference, swap_country_reference,
)

TARGETS = ["us", "netherlands", "hungary", "bosnia_and_herzegovina",
           "mexico", "taiwan", "philippines", "iran"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=2,
                    help="Example questions to print per form kind.")
    ap.add_argument("--kind", default=None,
                    help="Only print examples of this form kind.")
    ap.add_argument("--targets", default=",".join(TARGETS))
    args = ap.parse_args()

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")
    countries = sorted(set(ds["country"]))
    targets = [resolve_probe_country(t, countries) for t in args.targets.split(",")]

    kinds = collections.Counter()
    by_country = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    unswappable = []

    for r in ds:
        q, c = r["prompt_question"], r["country"]
        hit = find_country_reference(q, c)
        kind = hit[0] if hit else "NOT FOUND"
        kinds[kind] += 1
        by_country[c][kind] += 1
        examples[kind].append(r)
        if kind == "NOT FOUND":
            unswappable.append((c, q))

    print(f"rows {len(ds)}   countries {len(countries)}")
    print(f"\nreference form detected:")
    for k, v in kinds.most_common():
        print(f"  {k:12} {v:5d}  {v / len(ds):.3f}")

    # A target with no entry cannot be substituted in a demonym slot at all,
    # so it is a hard error rather than a coverage gap.
    bad_targets = [t for t in targets if t not in DEMONYMS]
    if bad_targets:
        print(f"\nTARGETS WITH NO DEMONYMS ENTRY: {bad_targets}")

    # Source countries need an entry only to be FOUND; a missing one shows up
    # as NOT FOUND rows, which is the number to drive down.
    missing = sorted({c for c in countries if c not in DEMONYMS})
    print(f"\nsource countries with no DEMONYMS entry: {missing or 'none'}")
    nf = sorted(((by_country[c]["NOT FOUND"], c) for c in countries), reverse=True)
    print("countries by unreferenced rows (these cannot be probed):")
    for n, c in nf:
        if n:
            print(f"  {c:26} {n:4d} / {sum(by_country[c].values()):4d}")

    print(f"\nper-country form breakdown:")
    for c in countries:
        print(f"  {c:26} {dict(by_country[c])}")

    for kind in (["NOT FOUND"] + [k for k in kinds if k != "NOT FOUND"]):
        if args.kind and kind != args.kind:
            continue
        print(f"\n{'=' * 72}\n{kind}  (n={kinds[kind]})")
        for r in examples[kind][:args.show]:
            print(f"\n  [{r['country']}] {r['prompt_question']}")
            if kind == "NOT FOUND":
                continue
            for t in targets:
                new, k = swap_country_reference(r["prompt_question"], r["country"], t)
                print(f"      {t:24} {new}")

    swappable = len(ds) - len(unswappable)
    print(f"\nswappable {swappable}/{len(ds)} = {swappable / len(ds):.3f}"
          f"   unreferenced {len(unswappable)}")

    deletable = sum(1 for r in ds
                    if strip_country_reference(r["prompt_question"], r["country"])[1])
    print(f"deletable (--no-country): {deletable}/{len(ds)} = {deletable / len(ds):.3f}")
    print("\nsample deletions:")
    shown = 0
    for r in ds:
        new, applied = strip_country_reference(r["prompt_question"], r["country"])
        if applied and shown < 6:
            print(f"  [{r['country']}] {r['prompt_question']}")
            print(f"  -> {new}\n")
            shown += 1

    return 1 if bad_targets else 0


if __name__ == "__main__":
    sys.exit(main())
