"""Extract the actual country/city/culture referenced in prompt text and save to datasets.

Uses spaCy NER to find GPE (countries/cities/states), NORP (nationalities/groups),
and LOC (geographic locations) entities. Saves a 'referenced_culture' field as a list
of unique entity strings found in the text.

For AYA datasets ('text' = inputs + '\n\n' + targets) only the inputs portion is scanned.
For DPO datasets the 'prompt' field is scanned.

Prerequisites:
    pip install spacy
    python -m spacy download en_core_web_lg

Usage:
    python scripts/extract_referenced_culture.py
    python scripts/extract_referenced_culture.py --data-dir data --model en_core_web_lg --batch-size 256
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENTITY_LABELS = {"GPE", "NORP", "LOC"}


def extract_entities(texts: list[str], nlp) -> list[list[str]]:
    results = []
    for doc in nlp.pipe(texts, batch_size=256, disable=["tagger", "parser", "lemmatizer"]):
        seen = []
        seen_lower = set()
        for ent in doc.ents:
            if ent.label_ in ENTITY_LABELS:
                text = ent.text.strip()
                if text.lower() not in seen_lower:
                    seen.append(text)
                    seen_lower.add(text.lower())
        results.append(seen)
    return results


def save_in_place(ds, path: Path) -> None:
    tmp    = path.parent / (path.name + "_tmp")
    backup = path.parent / (path.name + "_old")
    ds.save_to_disk(str(tmp))
    path.rename(backup)
    tmp.rename(path)
    shutil.rmtree(str(backup), ignore_errors=True)


def process_dataset(name: str, path: Path, nlp, text_field: str, aya_mode: bool) -> None:
    from datasets import load_from_disk

    print(f"\nProcessing {name}...")
    ds = load_from_disk(str(path))

    if "referenced_culture" in ds.column_names:
        print(f"  {name}: 'referenced_culture' already present, skipping")
        return

    def get_scan_text(ex):
        raw = ex[text_field]
        if aya_mode:
            # text = inputs + '\n\n' + targets — scan inputs portion only
            return raw.split("\n\n")[0]
        return raw

    texts = [get_scan_text(ex) for ex in ds]
    print(f"  Extracting entities from {len(texts):,} examples...")
    entity_lists = extract_entities(texts, nlp)

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
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--model",      default="en_core_web_lg",
                        help="spaCy model name (must be downloaded)")
    args = parser.parse_args()

    import spacy
    print(f"Loading spaCy model: {args.model}")
    nlp = spacy.load(args.model)

    data_dir = PROJECT_ROOT / args.data_dir

    datasets_cfg = [
        # (name,        text_field, aya_mode)
        ("aya_cult",  "text",   True),
        ("dpo_cult",  "prompt", False),
    ]

    for name, field, aya_mode in datasets_cfg:
        path = data_dir / name
        if not path.exists():
            print(f"\nSkipping {name}: not found at {path}")
            continue
        process_dataset(name, path, nlp, field, aya_mode)

    print("\nDone.")


if __name__ == "__main__":
    main()
