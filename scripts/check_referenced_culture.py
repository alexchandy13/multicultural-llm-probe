"""Sanity check: run spaCy NER + geo normalization on a small sample and print results.

Usage:
    python scripts/check_referenced_culture.py
    python scripts/check_referenced_culture.py --n 30
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTITY_LABELS = {"GPE", "NORP", "LOC"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n",        type=int, default=30)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    import spacy
    from extract_referenced_culture import (
        build_geo_lookup, _extract_raw, resolve_entities,
    )

    print("Loading en_core_web_lg...")
    nlp = spacy.load("en_core_web_lg")
    print("Building geo lookup...")
    country_names, city_to_country, extras, countries_by_code = build_geo_lookup()

    from datasets import load_from_disk
    data_dir = PROJECT_ROOT / args.data_dir

    rng = random.Random(args.seed)
    n_each = args.n // 2

    for name, field, aya_mode in [("aya_cult", "text", True), ("dpo_cult", "prompt", False)]:
        path = data_dir / name
        if not path.exists():
            print(f"\nSkipping {name}: not found")
            continue

        ds = load_from_disk(str(path))
        indices = rng.sample(range(len(ds)), min(n_each, len(ds)))
        sample = ds.select(indices)

        print(f"\n{'='*70}")
        print(f"  {name}  ({n_each} random examples)")
        print(f"{'='*70}")

        for ex in sample:
            raw_text = ex[field]
            scan_text = raw_text.split("\n\n")[0] if aya_mode else raw_text
            language   = ex.get("language", "")
            culture_tag = ex.get("culture_tag", "")

            doc = nlp(scan_text, disable=["tagger", "parser", "lemmatizer"])
            raw_ents = _extract_raw(doc)
            normalized = resolve_entities(raw_ents, country_names, city_to_country,
                                          extras, countries_by_code)

            display_text = scan_text[:120].replace("\n", " ")
            print(f"\n  [{culture_tag}] [{language}]")
            print(f"  text:       {display_text}{'...' if len(scan_text) > 120 else ''}")
            print(f"  raw NER:    {raw_ents if raw_ents else '(none)'}")
            print(f"  normalized: {normalized if normalized else '(none)'}")


if __name__ == "__main__":
    main()
