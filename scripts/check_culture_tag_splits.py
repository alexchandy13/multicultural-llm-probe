"""Show culture_tag distribution for aya_cult and dpo_cult datasets.

Usage:
    python scripts/check_culture_tag_splits.py
    python scripts/check_culture_tag_splits.py --data-dir data --out outputs/culture_tag_splits.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CULT_TAGS = [
    "CultureAsKnowledge",
    "CultureAsPreference",
    "CultureAsDynamics",
    "CultureAsBias",
    "GeneralCulture",
    "NoCulture",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=None, help="Save results to this JSON file")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    results = {}

    for name in ("aya_cult", "dpo_cult"):
        path = data_dir / name
        if not path.exists():
            print(f"\n{name}: not found at {path}")
            continue

        ds = load_from_disk(str(path))
        counts = Counter(ds["culture_tag"])
        total = len(ds)

        print(f"\n{name}  (n={total:,})")
        print(f"  {'tag':<25} {'count':>8}  {'%':>6}")
        print(f"  {'-'*42}")

        results[name] = {"total": total, "tags": {}}
        for tag in CULT_TAGS:
            n = counts.get(tag, 0)
            print(f"  {tag:<25} {n:>8,}  {n/total:>6.1%}")
            results[name]["tags"][tag] = {"count": n, "pct": round(n / total, 4)}

        other = {k: v for k, v in counts.items() if k not in CULT_TAGS}
        if other:
            print(f"  other: {other}")
            results[name]["other"] = other

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
