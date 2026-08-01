"""Extract the actual country/city/culture referenced in prompt text and save to datasets.

Uses spaCy en_core_web_lg NER to find GPE/NORP/LOC entities, then normalizes them:
  - Known country names are kept as-is
  - Known city names are mapped to their country
  - Nationality/demonym adjectives (NORP) are mapped to their country where possible
  - Anything not matching a known country or city is discarded

Saves a 'referenced_culture' field as a deduplicated list of country name strings.
Examples with no recognizable geographic reference get an empty list.

For AYA datasets ('text' = inputs + '\\n\\n' + targets) only the inputs portion is scanned.
For DPO datasets the 'prompt' field is scanned.

Prerequisites:
    pip install spacy geonamescache
    python -m spacy download en_core_web_lg

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

# Demonym/nationality → country name mappings not in geonamescache
DEMONYM_MAP = {
    "american": "United States", "american": "United States",
    "british":  "United Kingdom", "english": "United Kingdom",
    "welsh":    "United Kingdom", "scottish": "United Kingdom",
    "chinese":  "China", "japanese": "Japan", "korean": "South Korea",
    "indian":   "India", "russian":  "Russia", "french": "France",
    "german":   "Germany", "italian": "Italy", "spanish": "Spain",
    "mexican":  "Mexico", "canadian": "Canada", "australian": "Australia",
    "brazilian": "Brazil", "argentinian": "Argentina", "egyptian": "Egypt",
    "nigerian": "Nigeria", "south african": "South Africa",
    "saudi":    "Saudi Arabia", "emirati": "United Arab Emirates",
    "turkish":  "Turkey", "iranian":  "Iran", "iraqi": "Iraq",
    "pakistani": "Pakistan", "bangladeshi": "Bangladesh",
    "thai":     "Thailand", "vietnamese": "Vietnam",
    "indonesian": "Indonesia", "filipino": "Philippines",
    "dutch":    "Netherlands", "belgian": "Belgium", "swiss": "Switzerland",
    "swedish":  "Sweden", "norwegian": "Norway", "danish": "Denmark",
    "finnish":  "Finland", "polish":   "Poland", "greek": "Greece",
    "portuguese": "Portugal", "romanian": "Romania", "ukrainian": "Ukraine",
    "israeli":  "Israel", "lebanese": "Lebanon", "jordanian": "Jordan",
    "kenyan":   "Kenya", "ethiopian": "Ethiopia", "ghanaian": "Ghana",
    "colombian": "Colombia", "peruvian": "Peru", "chilean": "Chile",
    "venezuelan": "Venezuela", "cuban": "Cuba",
    "western":  "", "eastern": "", "southern": "", "northern": "",
    "european": "", "asian": "", "african": "", "latin": "",
    "middle eastern": "", "southeast asian": "",
}


def build_geo_lookup() -> tuple[set[str], dict[str, str]]:
    """Build known country name set and city→country dict from geonamescache."""
    import geonamescache
    gc = geonamescache.GeonamesCache()
    countries_by_code = gc.get_countries()

    country_names: set[str] = {c["name"].lower() for c in countries_by_code.values()}
    # also add common alternate names
    extras = {
        "usa": "United States", "us": "United States", "u.s.": "United States",
        "uk": "United Kingdom", "u.k.": "United Kingdom",
        "uae": "United Arab Emirates",
        "south korea": "South Korea", "north korea": "North Korea",
    }

    city_to_country: dict[str, str] = {}
    for city in gc.get_cities().values():
        cc = city["countrycode"]
        country = countries_by_code.get(cc, {}).get("name", "")
        if country:
            city_to_country[city["name"].lower()] = country

    return country_names, city_to_country, extras, countries_by_code


def normalize_entity(text: str, country_names: set[str],
                     city_to_country: dict[str, str],
                     extras: dict[str, str],
                     countries_by_code: dict) -> str | None:
    """Return normalized country name for entity text, or None to discard."""
    t = text.strip()
    tl = t.lower()

    # Direct country name match
    if tl in country_names:
        # Return the canonical casing from geonamescache
        for c in countries_by_code.values():
            if c["name"].lower() == tl:
                return c["name"]

    # Alternate/abbreviation match
    if tl in extras:
        return extras[tl] or None

    # City → country
    if tl in city_to_country:
        return city_to_country[tl]

    # Demonym/nationality adjective
    if tl in DEMONYM_MAP:
        return DEMONYM_MAP[tl] or None

    return None


def resolve_entities(raw_entities: list[str], country_names: set[str],
                     city_to_country: dict[str, str],
                     extras: dict[str, str],
                     countries_by_code: dict) -> list[str]:
    seen, seen_lower = [], set()
    for ent in raw_entities:
        country = normalize_entity(ent, country_names, city_to_country,
                                   extras, countries_by_code)
        if country and country.lower() not in seen_lower:
            seen.append(country)
            seen_lower.add(country.lower())
    return seen


def _extract_raw(doc) -> list[str]:
    seen, seen_lower = [], set()
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            t = ent.text.strip()
            if t.lower() not in seen_lower:
                seen.append(t)
                seen_lower.add(t.lower())
    return seen


def extract_and_normalize_batched(texts: list[str], nlp,
                                   country_names, city_to_country,
                                   extras, countries_by_code) -> list[list[str]]:
    results = []
    for doc in nlp.pipe(texts, batch_size=256, disable=["tagger", "parser", "lemmatizer"]):
        raw = _extract_raw(doc)
        results.append(resolve_entities(raw, country_names, city_to_country,
                                        extras, countries_by_code))
    return results


def save_in_place(ds, path: Path) -> None:
    tmp    = path.parent / (path.name + "_tmp")
    backup = path.parent / (path.name + "_old")
    ds.save_to_disk(str(tmp))
    path.rename(backup)
    tmp.rename(path)
    shutil.rmtree(str(backup), ignore_errors=True)


def process_dataset(name: str, path: Path, nlp, text_field: str, aya_mode: bool,
                    country_names, city_to_country, extras, countries_by_code) -> None:
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

    print(f"  Extracting + normalizing entities from {len(texts):,} examples...")
    entity_lists = extract_and_normalize_batched(
        texts, nlp, country_names, city_to_country, extras, countries_by_code)

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
    nlp = spacy.load("en_core_web_lg")

    print("Building geo lookup tables...")
    country_names, city_to_country, extras, countries_by_code = build_geo_lookup()
    print(f"  {len(country_names):,} countries, {len(city_to_country):,} cities indexed")

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
        process_dataset(name, path, nlp, field, aya_mode,
                        country_names, city_to_country, extras, countries_by_code)

    print("\nDone.")


if __name__ == "__main__":
    main()
