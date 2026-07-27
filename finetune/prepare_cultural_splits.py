"""Build culturally-split training sets by joining CultureMarkers tags with source datasets.

Streams CultureMarkers once to build {inputs_text: culture_tag} lookups for Aya,
UltraFeedback, and PRISM, then joins with the original datasets to recover responses.

Outputs (saved to --out-dir):
  aya_cult/    — Aya examples tagged as cultural (culture_tag != NoCulture), ~TARGET_SFT rows
  aya_nocult/  — Aya examples tagged as NoCulture, balanced to aya_cult size
  uf_cult/     — UltraFeedback preference pairs tagged as cultural
  uf_nocult/   — UltraFeedback preference pairs tagged as NoCulture, balanced to uf_cult
  prism_cult/  — PRISM preference pairs tagged as cultural, ~TARGET_DPO rows
  prism_nocult/— PRISM preference pairs tagged as NoCulture, balanced to prism_cult

The tag lookup dicts are cached as JSON so a re-run after partial failure skips
the ~30-min CultureMarkers stream. Delete tag_cache/ to force a re-stream.

Usage:
    python finetune/prepare_cultural_splits.py
    python finetune/prepare_cultural_splits.py --aya-only
    python finetune/prepare_cultural_splits.py --skip-prism
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset

TARGET_SFT = 52000   # match Alpaca (52k examples)
TARGET_DPO = 30000   # match HH-RLHF cap


def _norm(s: str) -> str:
    """Normalize whitespace for robust key matching across dataset versions."""
    return " ".join(s.split())


def build_tag_lookups(do_aya: bool, do_uf: bool, do_prism: bool,
                      cache_dir: Path) -> tuple[dict, dict, dict]:
    """Stream CultureMarkers once, build {inputs_text: culture_tag} for each source.

    Results are cached to cache_dir/*.json so a re-run after partial failure
    skips the expensive stream. Delete tag_cache/ to force a full re-stream.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    aya_cache   = cache_dir / "aya_tags.json"
    uf_cache    = cache_dir / "uf_tags.json"
    prism_cache = cache_dir / "prism_tags.json"

    aya_tags:   dict[str, str] = {}
    uf_tags:    dict[str, str] = {}
    prism_tags: dict[str, str] = {}

    def _load_cache(path, tags):
        with open(path) as f:
            tags.update(json.load(f))
        print(f"  loaded {len(tags):,} entries from {path}")

    need_aya   = do_aya   and not aya_cache.exists()
    need_uf    = do_uf    and not uf_cache.exists()
    need_prism = do_prism and not prism_cache.exists()

    # Load whatever is already cached
    if do_aya   and not need_aya:   _load_cache(aya_cache,   aya_tags)
    if do_uf    and not need_uf:    _load_cache(uf_cache,    uf_tags)
    if do_prism and not need_prism: _load_cache(prism_cache, prism_tags)

    if need_aya or need_uf or need_prism:
        print("Streaming CultureMarkers to build tag lookups (5.6M rows, ~30 min)...")
        ds = load_dataset("CohereLabs/CultureMarkers", split="train", streaming=True)
        total = 0
        for ex in ds:
            src = ex["dataset_source"]
            if need_aya and src == "Aya":
                aya_tags[_norm(ex["inputs"])] = ex["culture_tag"]
            elif need_uf and src == "UltraFeedback-Alignment":
                uf_tags[_norm(ex["inputs"])] = ex["culture_tag"]
            elif need_prism and src == "Prism-Alignment":
                prism_tags[_norm(ex["inputs"])] = ex["culture_tag"]
            total += 1
            if total % 200_000 == 0:
                print(f"  {total:,} rows | aya={len(aya_tags):,} "
                      f"| uf={len(uf_tags):,} | prism={len(prism_tags):,}")
        print(f"Done. aya={len(aya_tags):,} | uf={len(uf_tags):,} | prism={len(prism_tags):,}")

        if need_aya:
            with open(aya_cache, "w") as f: json.dump(aya_tags, f)
            print(f"Cached aya_tags -> {aya_cache}")
        if need_uf:
            with open(uf_cache, "w") as f: json.dump(uf_tags, f)
            print(f"Cached uf_tags -> {uf_cache}")
        if need_prism:
            with open(prism_cache, "w") as f: json.dump(prism_tags, f)
            print(f"Cached prism_tags -> {prism_cache}")

    return aya_tags, uf_tags, prism_tags


def prepare_aya(aya_tags: dict[str, str], out_dir: Path, seed: int = 42) -> None:
    """Join Aya dataset with culture tags, save cultural/nocult splits as SFT text."""
    print("\nLoading Aya dataset...")
    ds = load_dataset("CohereForAI/aya_dataset", split="train")

    cult_rows:   list[dict] = []
    nocult_rows: list[dict] = []
    n_matched = 0

    for ex in ds:
        tag = aya_tags.get(_norm(ex["inputs"]))
        if tag is None:
            continue
        n_matched += 1
        row = {"text": f"{ex['inputs']}\n\n{ex['targets']}"}
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
    nocult = Dataset.from_list(nocult_rows[:len(cult)])
    nocult.save_to_disk(str(out_dir / "aya_nocult"))
    print(f"Saved aya_nocult: {len(nocult):,} examples")


def prepare_ultrafeedback(uf_tags: dict[str, str], out_dir: Path, seed: int = 42) -> None:
    """Join UltraFeedback binarized with culture tags, save cultural/nocult preference pairs."""
    print("\nLoading UltraFeedback binarized (train_prefs)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")

    cult_rows:   list[dict] = []
    nocult_rows: list[dict] = []
    n_matched = 0

    for ex in ds:
        tag = uf_tags.get(_norm(ex["prompt"]))
        if tag is None:
            continue
        n_matched += 1
        row = {
            "prompt":   ex["prompt"],
            "chosen":   ex["chosen"][-1]["content"],
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
    nocult = Dataset.from_list(nocult_rows[:len(cult)])
    nocult.save_to_disk(str(out_dir / "uf_nocult"))
    print(f"Saved uf_nocult: {len(nocult):,} examples")


def prepare_prism(prism_tags: dict[str, str], out_dir: Path, seed: int = 42) -> None:
    """Join PRISM utterances with culture tags, save cultural/nocult preference pairs.

    Groups utterances by (conversation_id, turn), picks if_chosen=True as chosen
    and the lowest-scored other response as rejected.
    """
    print("\nLoading PRISM utterances...")
    ds = load_dataset("HannahRoseKirk/prism-alignment", "utterances", split="train")

    # Group by (conversation_id, turn) -> list of rows
    groups: dict[tuple, list] = defaultdict(list)
    for ex in ds:
        groups[(ex["conversation_id"], ex["turn"])].append(ex)

    cult_rows:   list[dict] = []
    nocult_rows: list[dict] = []
    n_matched = 0

    for rows in groups.values():
        chosen_rows   = [r for r in rows if r["if_chosen"]]
        rejected_rows = [r for r in rows if not r["if_chosen"]]
        if not chosen_rows or not rejected_rows:
            continue

        prompt = chosen_rows[0]["user_prompt"]
        tag = prism_tags.get(_norm(prompt))
        if tag is None:
            continue
        n_matched += 1

        # pick lowest-scored rejected to maximize preference signal
        rejected = min(rejected_rows, key=lambda r: r["score"])
        row = {
            "prompt":   prompt,
            "chosen":   chosen_rows[0]["model_response"],
            "rejected": rejected["model_response"],
        }
        if tag == "NoCulture":
            nocult_rows.append(row)
        else:
            cult_rows.append(row)

    print(f"PRISM: {n_matched:,} matched | cultural={len(cult_rows):,} | nocult={len(nocult_rows):,}")

    rng = random.Random(seed)
    rng.shuffle(cult_rows)
    cult = Dataset.from_list(cult_rows[:TARGET_DPO])
    cult.save_to_disk(str(out_dir / "prism_cult"))
    print(f"Saved prism_cult: {len(cult):,} examples")

    rng.shuffle(nocult_rows)
    nocult = Dataset.from_list(nocult_rows[:len(cult)])
    nocult.save_to_disk(str(out_dir / "prism_nocult"))
    print(f"Saved prism_nocult: {len(nocult):,} examples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data", help="Root dir for output splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aya-only",   action="store_true", help="Only run Aya")
    parser.add_argument("--uf-only",    action="store_true", help="Only run UltraFeedback")
    parser.add_argument("--prism-only", action="store_true", help="Only run PRISM")
    parser.add_argument("--skip-prism", action="store_true", help="Skip PRISM")
    args = parser.parse_args()

    explicit = args.aya_only or args.uf_only or args.prism_only
    do_aya   = args.aya_only   or not explicit
    do_uf    = args.uf_only    or not explicit
    do_prism = args.prism_only or (not explicit and not args.skip_prism)

    out_dir   = Path(args.out_dir)
    cache_dir = out_dir / "tag_cache"

    aya_tags, uf_tags, prism_tags = build_tag_lookups(do_aya, do_uf, do_prism, cache_dir)

    if do_aya:
        prepare_aya(aya_tags, out_dir, args.seed)
    if do_uf:
        prepare_ultrafeedback(uf_tags, out_dir, args.seed)
    if do_prism:
        prepare_prism(prism_tags, out_dir, args.seed)

    print("\nDone. All splits saved.")


if __name__ == "__main__":
    main()
