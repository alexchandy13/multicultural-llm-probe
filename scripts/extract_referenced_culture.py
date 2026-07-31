"""Extract the actual country/city/culture referenced in prompt text and save to datasets.

Uses spaCy NER to find GPE (countries/cities/states), NORP (nationalities/groups),
and LOC (geographic locations) entities. Saves a 'referenced_culture' field as a list
of unique entity strings found in the text.

Routes by the 'language' field already in each dataset:
  - English examples  → en_core_web_lg  (higher accuracy)
  - All other languages → xx_ent_wiki_sm (multilingual, ~40 languages)

For AYA datasets ('text' = inputs + '\n\n' + targets) only the inputs portion is scanned.
For DPO datasets the 'prompt' field is scanned.

Prerequisites:
    pip install spacy
    python -m spacy download en_core_web_lg
    python -m spacy download xx_ent_wiki_sm

Usage:
    python scripts/extract_referenced_culture.py
    python scripts/extract_referenced_culture.py --data-dir data
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENTITY_LABELS = {"GPE", "NORP", "LOC"}


def _dedupe_entities(doc) -> list[str]:
    seen, seen_lower = [], set()
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            t = ent.text.strip()
            if t.lower() not in seen_lower:
                seen.append(t)
                seen_lower.add(t.lower())
    return seen


def extract_entities_batched(texts: list[str], nlp_en, nlp_xx) -> list[list[str]]:
    """Run en_core_web_lg on all texts; fall back to xx_ent_wiki_sm where nothing is found."""
    results: list[list[str]] = [[] for _ in texts]

    # Pass 1: English model on everything
    for i, doc in enumerate(nlp_en.pipe(
            texts, batch_size=256, disable=["tagger", "parser", "lemmatizer"])):
        results[i] = _dedupe_entities(doc)

    # Pass 2: multilingual model only where English found nothing
    fallback_indices = [i for i, r in enumerate(results) if not r]
    if fallback_indices:
        fallback_texts = [texts[i] for i in fallback_indices]
        for batch_idx, doc in enumerate(nlp_xx.pipe(
                fallback_texts, batch_size=256, disable=["tagger", "parser", "lemmatizer"])):
            results[fallback_indices[batch_idx]] = _dedupe_entities(doc)

    return results


def save_in_place(ds, path: Path) -> None:
    tmp    = path.parent / (path.name + "_tmp")
    backup = path.parent / (path.name + "_old")
    ds.save_to_disk(str(tmp))
    path.rename(backup)
    tmp.rename(path)
    shutil.rmtree(str(backup), ignore_errors=True)


def process_dataset(name: str, path: Path, nlp_en, nlp_xx,
                    text_field: str, aya_mode: bool) -> None:
    from datasets import load_from_disk

    print(f"\nProcessing {name}...")
    ds = load_from_disk(str(path))

    if "referenced_culture" in ds.column_names:
        print(f"  {name}: 'referenced_culture' already present, skipping")
        return

    texts = []
    for ex in ds:
        raw = ex[text_field]
        texts.append(raw.split("\n\n")[0] if aya_mode else raw)

    print(f"  Pass 1: en_core_web_lg on all {len(texts):,} examples...")
    entity_lists = extract_entities_batched(texts, nlp_en, nlp_xx)
    n_fallback = sum(1 for e in entity_lists if not e)
    print(f"  Pass 2: xx_ent_wiki_sm fallback on {n_fallback:,} examples with no entities found")

    n_with_culture = sum(1 for e in entity_lists if e)
    print(f"  {n_with_culture:,}/{len(texts):,} examples have at least one entity "
          f"({n_with_culture/len(texts):.1%})")

    def _add(ex, idx):
        return {"referenced_culture": entity_lists[idx]}

    ds = ds.map(_add, with_indices=True, desc=f"Adding referenced_culture to {name}")
    save_in_place(ds, path)
    print(f"  Saved {name}: {len(ds):,} examples, columns: {ds.column_names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    import spacy
    print("Loading en_core_web_lg...")
    nlp_en = spacy.load("en_core_web_lg")
    print("Loading xx_ent_wiki_sm...")
    nlp_xx = spacy.load("xx_ent_wiki_sm")

    data_dir = PROJECT_ROOT / args.data_dir

    datasets_cfg = [
        ("aya_cult", "text",   True),
        ("dpo_cult", "prompt", False),
    ]

    for name, field, aya_mode in datasets_cfg:
        path = data_dir / name
        if not path.exists():
            print(f"\nSkipping {name}: not found at {path}")
            continue
        process_dataset(name, path, nlp_en, nlp_xx, field, aya_mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
