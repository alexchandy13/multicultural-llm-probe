"""Add culture_tag, domain_tag, task_intent_tag, and language to existing curated datasets.

Rebuilds rich tag caches by re-streaming CultureMarkers (skipped if already rich),
then joins with each saved dataset to add metadata columns without changing examples.

For AYA datasets the saved 'text' field is '{inputs}\\n\\n{targets}', but the cache
is keyed on _norm(inputs). We re-stream the AYA source dataset to build a
{_norm(full_text): metadata} lookup so the join works correctly.

Usage:
    python scripts/add_metadata_to_splits.py
    python scripts/add_metadata_to_splits.py --data-dir data
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _norm(s: str) -> str:
    return " ".join(s.split())


def _is_rich(tags: dict) -> bool:
    if not tags:
        return False
    return isinstance(next(iter(tags.values())), dict)


def build_rich_cache(cache_dir: Path) -> tuple[dict, dict, dict]:
    """Build or load rich {inputs_text: tag_dict} caches from CultureMarkers."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "aya":   cache_dir / "aya_tags.json",
        "uf":    cache_dir / "uf_tags.json",
        "prism": cache_dir / "prism_tags.json",
    }
    tags: dict[str, dict] = {"aya": {}, "uf": {}, "prism": {}}

    need: dict[str, bool] = {}
    for src, path in paths.items():
        if path.exists():
            with open(path) as f:
                loaded = json.load(f)
            if _is_rich(loaded):
                tags[src] = loaded
                print(f"  Loaded rich cache: {path} ({len(loaded):,} entries)")
                need[src] = False
            else:
                print(f"  Old-format cache at {path} — will re-stream")
                need[src] = True
        else:
            need[src] = True

    if any(need.values()):
        src_map = {"aya": "Aya", "uf": "UltraFeedback-Alignment", "prism": "Prism-Alignment"}
        need_srcs = {src_map[k] for k, v in need.items() if v}
        print(f"Streaming CultureMarkers for: {need_srcs} (~30 min)...")
        ds = load_dataset("CohereLabs/CultureMarkers", split="train", streaming=True)
        total = 0
        for ex in ds:
            src = ex["dataset_source"]
            if src not in need_srcs:
                continue
            key = _norm(ex["inputs"])
            tag_dict = {
                "culture_tag":     ex.get("culture_tag", ""),
                "domain_tag":      ex.get("domain_tag", ""),
                "task_intent_tag": ex.get("task_intent_tag", ""),
                "language":        ex.get("language", ""),
            }
            if src == "Aya" and need.get("aya"):
                tags["aya"][key] = tag_dict
            elif src == "UltraFeedback-Alignment" and need.get("uf"):
                tags["uf"][key] = tag_dict
            elif src == "Prism-Alignment" and need.get("prism"):
                tags["prism"][key] = tag_dict
            total += 1
            if total % 200_000 == 0:
                print(f"  {total:,} rows streamed")

        for src, should_write in need.items():
            if should_write:
                with open(paths[src], "w") as f:
                    json.dump(tags[src], f)
                print(f"  Saved rich cache: {paths[src]} ({len(tags[src]):,} entries)")

    return tags["aya"], tags["uf"], tags["prism"]


def build_aya_text_lookup(aya_cache: dict) -> dict:
    """Build {_norm(full_text): metadata} by joining AYA source dataset with cache.

    Needed because saved AYA rows store '{inputs}\\n\\n{targets}' as 'text',
    but the cache is keyed on _norm(inputs) only.
    """
    print("  Building AYA full-text lookup (streaming aya_dataset)...")
    ds = load_dataset("CohereForAI/aya_dataset", split="train")
    lookup: dict[str, dict] = {}
    n_matched = 0
    for ex in ds:
        key = _norm(ex["inputs"])
        if key in aya_cache:
            full_text_key = _norm(f"{ex['inputs']}\n\n{ex['targets']}")
            lookup[full_text_key] = aya_cache[key]
            n_matched += 1
    print(f"  AYA lookup: {n_matched:,} entries built from {len(ds):,} source examples")
    return lookup


def add_metadata(ds, cache: dict, key_field: str, name: str):
    """Add metadata columns to dataset by joining on key_field."""
    meta_fields = ["culture_tag", "domain_tag", "task_intent_tag", "language"]
    if all(f in ds.column_names for f in meta_fields):
        print(f"  {name}: metadata already present, skipping")
        return ds

    n_matched = 0
    def _add(ex):
        nonlocal n_matched
        key = _norm(ex.get(key_field, ""))
        tags = cache.get(key, {})
        if tags:
            n_matched += 1
        return {
            "culture_tag":     tags.get("culture_tag", ""),
            "domain_tag":      tags.get("domain_tag", ""),
            "task_intent_tag": tags.get("task_intent_tag", ""),
            "language":        tags.get("language", ""),
        }

    ds = ds.map(_add, desc=f"Adding metadata to {name}")
    print(f"  {name}: matched {n_matched:,}/{len(ds):,} examples")
    return ds


def save_in_place(ds, path: Path) -> None:
    """Save dataset to path, working around the can't-overwrite-itself restriction.

    Uses rename-aside strategy to avoid NFS rmtree issues with lingering .nfs* files.
    """
    tmp    = path.parent / (path.name + "_tmp")
    backup = path.parent / (path.name + "_old")
    ds.save_to_disk(str(tmp))
    path.rename(backup)
    tmp.rename(path)
    shutil.rmtree(str(backup), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir  = PROJECT_ROOT / args.data_dir
    cache_dir = data_dir / "tag_cache"

    aya_cache, uf_cache, prism_cache = build_rich_cache(cache_dir)
    dpo_cache = {**uf_cache, **prism_cache}

    # AYA needs a full-text lookup because saved 'text' = inputs + '\n\n' + targets
    aya_text_lookup = build_aya_text_lookup(aya_cache)

    datasets_cfg = [
        ("aya_cult",   aya_text_lookup, "text"),
        ("aya_nocult", aya_text_lookup, "text"),
        ("dpo_cult",   dpo_cache,       "prompt"),
        ("dpo_nocult", dpo_cache,       "prompt"),
    ]

    for name, cache, key_field in datasets_cfg:
        path = data_dir / name
        if not path.exists():
            print(f"\nSkipping {name}: not found at {path}")
            continue
        print(f"\nProcessing {name}...")
        ds = load_from_disk(str(path))
        ds = add_metadata(ds, cache, key_field, name)
        save_in_place(ds, path)
        print(f"  Saved {name}: {len(ds):,} examples, columns: {ds.column_names}")


if __name__ == "__main__":
    main()
