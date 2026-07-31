"""Show culture_tag, domain_tag, task_intent_tag, and language distributions
for the 4 curated datasets (aya_cult, aya_nocult, dpo_cult, dpo_nocult).

If the saved datasets already contain these fields (new format), reads them directly.
Otherwise falls back to joining via the tag cache built by prepare_cultural_splits.py.

Usage:
    python scripts/analyze_data_distribution.py
    python scripts/analyze_data_distribution.py --data-dir data --top 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    return " ".join(s.split())


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path) as f:
        data = json.load(f)
    sample = next(iter(data.values()), None) if data else None
    if sample is None or not isinstance(sample, dict):
        print(f"  Warning: {cache_path} is old-format (plain strings). Re-run prepare_cultural_splits.py to rebuild.")
        return {}
    return data


def show_dist(counter: Counter, label: str, top: int) -> None:
    total = sum(counter.values())
    if total == 0:
        print(f"  {label}: no data")
        return
    print(f"  {label} (top {top}, total={total:,}):")
    for val, cnt in counter.most_common(top):
        print(f"    {val or '(empty)':<40} {cnt:>6,}  {cnt/total:>6.1%}")


def analyze_dataset(name: str, ds, tag_cache: dict, key_field: str, top: int) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}  ({len(ds):,} examples)")
    print(f"{'='*60}")

    fields = ["culture_tag", "domain_tag", "task_intent_tag", "language"]
    has_fields = all(f in ds.column_names for f in fields)

    counters = {f: Counter() for f in fields}

    if has_fields:
        for ex in ds:
            for f in fields:
                counters[f][ex[f]] += 1
    elif tag_cache:
        print("  (metadata not in dataset — joining via tag cache)")
        for ex in ds:
            key = _norm(ex.get(key_field, ""))
            tags = tag_cache.get(key, {})
            for f in fields:
                counters[f][tags.get(f, "")] += 1
    else:
        print("  No metadata fields and no tag cache available.")
        print("  Re-run prepare_cultural_splits.py to rebuild caches in rich format.")
        return

    for f in fields:
        show_dist(counters[f], f, top)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    data_dir  = PROJECT_ROOT / args.data_dir
    cache_dir = data_dir / "tag_cache"

    aya_cache   = load_cache(cache_dir / "aya_tags.json")
    uf_cache    = load_cache(cache_dir / "uf_tags.json")
    prism_cache = load_cache(cache_dir / "prism_tags.json")
    dpo_cache   = {**uf_cache, **prism_cache}

    datasets_cfg = [
        ("aya_cult",   "aya_cult",   aya_cache,  "text"),
        ("aya_nocult", "aya_nocult", aya_cache,  "text"),
        ("dpo_cult",   "dpo_cult",   dpo_cache,  "prompt"),
        ("dpo_nocult", "dpo_nocult", dpo_cache,  "prompt"),
    ]

    for name, subdir, cache, key_field in datasets_cfg:
        path = data_dir / subdir
        if not path.exists():
            print(f"\nSkipping {name}: not found at {path}")
            continue
        ds = load_from_disk(str(path))
        analyze_dataset(name, ds, cache, key_field, args.top)


if __name__ == "__main__":
    main()
