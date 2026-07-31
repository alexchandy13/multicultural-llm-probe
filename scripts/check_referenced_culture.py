"""Sanity check: run spaCy NER on a small sample and print results.

Routes by the 'language' field: English → en_core_web_lg, other → xx_ent_wiki_sm.

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


def extract_entities(text: str, nlp) -> list[str]:
    doc = nlp(text, disable=["tagger", "parser", "lemmatizer"])
    seen, seen_lower = [], set()
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            t = ent.text.strip()
            if t.lower() not in seen_lower:
                seen.append(t)
                seen_lower.add(t.lower())
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n",        type=int, default=30)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    import spacy
    print("Loading en_core_web_lg...")
    nlp_en = spacy.load("en_core_web_lg")
    print("Loading xx_ent_wiki_sm...")
    nlp_xx = spacy.load("xx_ent_wiki_sm")

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
            raw = ex[field]
            scan_text = raw.split("\n\n")[0] if aya_mode else raw
            language = ex.get("language", "")
            entities = extract_entities(scan_text, nlp_en)
            model_used = "en"
            if not entities:
                entities = extract_entities(scan_text, nlp_xx)
                model_used = "xx" if entities else "en"
            culture_tag = ex.get("culture_tag", "")
            display_text = scan_text[:120].replace("\n", " ")
            print(f"\n  [{culture_tag}] [{language}] [model={model_used}]")
            print(f"  text:     {display_text}{'...' if len(scan_text) > 120 else ''}")
            print(f"  entities: {entities if entities else '(none)'}")


if __name__ == "__main__":
    main()
