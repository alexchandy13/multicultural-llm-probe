"""Build culturally-split training sets by joining CultureMarkers tags with source datasets.

Streams CultureMarkers once to build {inputs_text: culture_tag} lookups for Aya
and UltraFeedback, then joins with the original datasets to recover responses.

Outputs (saved to --out-dir):
  aya_cult/    — Aya examples tagged as cultural (culture_tag != NoCulture), ~TARGET_SFT rows
  aya_nocult/  — Aya examples tagged as NoCulture, ~TARGET_SFT rows
  uf_cult/     — UltraFeedback preference pairs tagged as cultural, ~TARGET_DPO rows
  uf_nocult/   — UltraFeedback preference pairs tagged as NoCulture, ~TARGET_DPO rows

The tag lookup dicts are cached as JSON so a re-run after partial failure skips
the 30-min CultureMarkers stream. Delete tag_cache/ to force a re-stream.

Usage:
    python data/prepare_cultural_splits.py
    python data/prepare_cultural_splits.py --aya-only   # skip UltraFeedback
    python data/prepare_cultural_splits.py --uf-only    # skip Aya
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

TARGET_SFT = 52000   # match Alpaca (52k examples)
TARGET_DPO = 30000   # match HH-RLHF cap


def build_tag_lookups(do_aya: bool, do_uf: bool, cache_dir: Path) -> tuple[dict, dict]:
    """Stream CultureMarkers once, build {inputs_text: culture_tag} for Aya and UF.

    Results are cached to cache_dir/aya_tags.json and uf_tags.json so a re-run
    after a partial failure skips the expensive stream.
    """
    aya_cache = cache_dir / "aya_tags.json"
    uf_cache = cache_dir / "uf_tags.json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    aya_tags: dict[str, str] = {}
    uf_tags: dict[str, str] = {}

    need_stream = (do_aya and not aya_cache.exists()) or (do_uf and not uf_cache.exists())

    if not need_stream:
        if do_aya:
            print(f"Loading aya_tags from cache ({aya_cache})...")
            with open(aya_cache) as f:
                aya_tags = json.load(f)
            print(f"  aya_tags: {len(aya_tags):,} entries")
        if do_uf:
            print(f"Loading uf_tags from cache ({uf_cache})...")
            with open(uf_cache) as f:
                uf_tags = json.load(f)
            print(f"  uf_tags: {len(uf_tags):,} entries")
        return aya_tags, uf_tags

    # Load existing cache for whichever side is done
    if do_aya and aya_cache.exists():
        with open(aya_cache) as f:
            aya_tags = json.load(f)
        print(f"aya_tags already cached ({len(aya_tags):,}), will skip during stream")
        do_aya_stream = False
    else:
        do_aya_stream = do_aya

    if do_uf and uf_cache.exists():
        with open(uf_cache) as f:
            uf_tags = json.load(f)
        print(f"uf_tags already cached ({len(uf_tags):,}), will skip during stream")
        do_uf_stream = False
    else:
        do_uf_stream = do_uf

    if do_aya_stream or do_uf_stream:
        print("Streaming CultureMarkers to build tag lookups (5.6M rows, ~30 min)...")
        ds = load_dataset("CohereLabs/CultureMarkers", split="train", streaming=True)
        total = 0
        for ex in ds:
            src = ex["dataset_source"]
            if do_aya_stream and src == "Aya":
                aya_tags[ex["inputs"]] = ex["culture_tag"]
            elif do_uf_stream and src == "UltraFeedback-Alignment":
                uf_tags[ex["inputs"]] = ex["culture_tag"]
            total += 1
            if total % 200_000 == 0:
                print(f"  {total:,} rows scanned | aya={len(aya_tags):,} | uf={len(uf_tags):,}")
        print(f"Done. aya={len(aya_tags):,} entries | uf={len(uf_tags):,} entries")

        if do_aya_stream:
            with open(aya_cache, "w") as f:
                json.dump(aya_tags, f)
            print(f"Cached aya_tags -> {aya_cache}")
        if do_uf_stream:
            with open(uf_cache, "w") as f:
                json.dump(uf_tags, f)
            print(f"Cached uf_tags -> {uf_cache}")

    return aya_tags, uf_tags


def prepare_aya(aya_tags: dict[str, str], out_dir: Path, seed: int = 42) -> None:
    """Join Aya dataset with culture tags, save cultural/nocult splits as SFT text."""
    print("\nLoading Aya dataset...")
    ds = load_dataset("CohereForAI/aya_dataset", split="train")

    cult_rows: list[dict] = []
    nocult_rows: list[dict] = []
    n_matched = 0

    for ex in ds:
        tag = aya_tags.get(ex["inputs"])
        if tag is None:
            continue
        n_matched += 1
        text = f"{ex['inputs']}\n\n{ex['targets']}"
        row = {"text": text}
        if tag == "NoCulture":
            nocult_rows.append(row)
        else:
            cult_rows.append(row)

    print(f"Aya: {n_matched:,} matched | cultural={len(cult_rows):,} | nocult={len(nocult_rows):,}")

    rng = random.Random(seed)

    rng.shuffle(cult_rows)
    cult = Dataset.from_list(cult_rows[:TARGET_SFT])
    cult.save_to_disk(str(out_dir / "aya_cult"))
    print(f"Saved aya_cult: {len(cult):,} examples")

    rng.shuffle(nocult_rows)
    nocult = Dataset.from_list(nocult_rows[:TARGET_SFT])
    nocult.save_to_disk(str(out_dir / "aya_nocult"))
    print(f"Saved aya_nocult: {len(nocult):,} examples")


def prepare_ultrafeedback(uf_tags: dict[str, str], out_dir: Path, seed: int = 42) -> None:
    """Join UltraFeedback binarized with culture tags, save cultural/nocult preference pairs."""
    print("\nLoading UltraFeedback binarized (train_prefs)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    cult_rows: list[dict] = []
    nocult_rows: list[dict] = []
    n_matched = 0

    for ex in ds:
        tag = uf_tags.get(ex["prompt"])
        if tag is None:
            continue
        n_matched += 1
        # chosen/rejected are lists of messages; last element is the assistant turn
        row = {
            "prompt": ex["prompt"],
            "chosen": ex["chosen"][-1]["content"],
            "rejected": ex["rejected"][-1]["content"],
        }
        if tag == "NoCulture":
            nocult_rows.append(row)
        else:
            cult_rows.append(row)

    print(f"UF: {n_matched:,} matched | cultural={len(cult_rows):,} | nocult={len(nocult_rows):,}")

    rng = random.Random(seed)

    rng.shuffle(cult_rows)
    cult = Dataset.from_list(cult_rows[:TARGET_DPO])
    cult.save_to_disk(str(out_dir / "uf_cult"))
    print(f"Saved uf_cult: {len(cult):,} examples")

    rng.shuffle(nocult_rows)
    nocult = Dataset.from_list(nocult_rows[:TARGET_DPO])
    nocult.save_to_disk(str(out_dir / "uf_nocult"))
    print(f"Saved uf_nocult: {len(nocult):,} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data", help="Root dir for output splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aya-only", action="store_true", help="Skip UltraFeedback")
    parser.add_argument("--uf-only", action="store_true", help="Skip Aya")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "tag_cache"
    do_aya = not args.uf_only
    do_uf = not args.aya_only

    aya_tags, uf_tags = build_tag_lookups(do_aya, do_uf, cache_dir)

    if do_aya:
        prepare_aya(aya_tags, out_dir, args.seed)
    if do_uf:
        prepare_ultrafeedback(uf_tags, out_dir, args.seed)

    print("\nDone. All splits saved.")


if __name__ == "__main__":
    main()
