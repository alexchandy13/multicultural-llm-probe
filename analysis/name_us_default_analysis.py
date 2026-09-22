"""Analyzes whether US defaulting rate in NormAd correlates with Anglo/English names.

For each NormAd example, extracts person names using spaCy NER and classifies
them as Anglo (English-speaking-world names) or non-Anglo. Then computes US default
rate (fraction of errors matching the US probe prediction) by name type.

Usage:
    python analysis/name_us_default_analysis.py
    python analysis/name_us_default_analysis.py --model gemma4
    python analysis/name_us_default_analysis.py --condition base
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"

HOLDOUT_COUNTRIES = {"Syria", "Indonesia", "Colombia", "Austria", "north_macedonia", "Sweden"}


def gold_label(ex: dict) -> str:
    return str(ex.get("Gold Label", ex.get("gold_label", ""))).lower().strip()


def scenario_text(ex: dict) -> str:
    for key in ("Story", "story", "scenario", "situation", "text"):
        if key in ex and ex[key]:
            return str(ex[key]).strip()
    return ""


ANGLO_NAMES = {
    "Sarah", "Emily", "Alex", "Michael", "Alice", "Tom", "Mike", "Sam",
    "Chris", "Mark", "Emma", "Ben", "Jamie", "Kevin", "Samantha", "Sara",
    "James", "Bob", "Liam", "Thomas", "Daniel", "Lily", "Jake", "Charlie",
    "David", "Tim", "Jack", "Peter", "Ethan", "Max", "Lucas", "Jordan",
    "Lucy", "Jane", "Susan", "Rachel", "Sophie", "Sophia", "Linda", "John",
    "Noah", "Luca", "Clara", "Anna", "Lisa", "Mia", "Maria",
    "Henry", "Charlotte", "George", "William", "Oliver",
    "Chloe", "Nathan", "Ryan", "Andrew", "Luke", "Aaron", "Adam",
    "Amy", "Kate", "Jessica", "Laura", "Elizabeth", "Eric", "Paul",
    "Lena", "Taylor", "Dan", "Leah", "Zoe", "Nina", "Megan", "Maya",
    "Tina", "Hannah", "Martin", "Jacob", "Claire", "Ella", "Ava",
    "Jenna", "Ann", "Isaac", "Olivia", "Ellie", "Eva", "Jenny",
    "Michelle", "Elisa", "Brian", "Jeff", "Josh", "Cameron", "Erin",
    "Sally", "Dana", "Eddie", "Stephen", "Charles", "Leo", "Steve",
    "Sandra", "Alan", "Lee", "Elliot", "Grace", "Julie", "Pat",
    "Andy", "Anne", "Aiden", "Jerry", "Gary", "Kim", "Steven",
    "Marcus", "Anton", "Julian", "Amelia", "June", "Benjamin", "Alicia",
    "Sophie", "Tina", "Marcus",
}

NON_ANGLO_NAMES = {
    # Arabic/Persian/Islamic
    "Amir", "Amina", "Layla", "Ahmed", "Fatima", "Hassan", "Mohammad",
    "Leila", "Omar", "Yasmin", "Yusuf", "Nadia", "Rania", "Tariq",
    "Khalid", "Samir", "Ali", "Aisha", "Jamal", "Aliyah", "Arif",
    "Ahmad", "Leyla", "Asli", "Alem",
    # South Asian
    "Priya", "Raj", "Ananya", "Vikram", "Deepa", "Arjun", "Kavya", "Ravi",
    "Rahul", "Arjuna", "Neeta", "Meera", "Riya", "Hari", "Sita",
    # East Asian
    "Mei", "Ming", "Wei", "Zhang", "Chen", "Li", "Yuki", "Kenji", "Hana",
    "Jin", "Soo", "Park", "Yuna", "Hiroshi", "Akira", "Mai",
    "Linh", "Niran", "Noy", "Mali", "Minh", "Jiho",
    # Southeast Asian / Pacific
    "Aung", "Sione", "Alani",
    # African / Sub-Saharan
    "Thandi", "Tendai", "Dara", "Amara", "Kofi", "Chioma", "Lindiwe",
    "Sipho", "Tadesse", "Nyasha", "Lulit", "Dawit", "Dira",
    "Thabo", "Lerato", "Amani",
    # Slavic/Eastern European
    "Marko", "Olena", "Dmitri", "Natasha", "Ivan", "Anastasia",
    "Mykola", "Luka", "Lukas", "Andrei", "Marta", "Marek", "Jasmina",
    # Spanish/Latin
    "Carlos", "Elena", "Sofia", "Ana", "Miguel", "Pedro", "Rosa",
    "Luis", "Mira", "Lina", "Manuel", "Lucia", "Alejandro", "Ayesha",
    # Scandinavian / Germanic
    "Johan", "Lars", "Erik", "Andreas", "Markus",
    # Other
    "Mila", "Arlin",
}


def _load_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


SKIP_WORDS = {
    # Pronouns and articles
    "He", "She", "They", "It", "His", "Her", "This", "The", "As",
    # Prepositions / conjunctions
    "At", "In", "On", "To", "By", "So", "Or", "Not", "Due",
    "After", "Before", "During", "While", "When", "Upon", "Once",
    "Without", "Instead", "Despite", "Although", "Though", "Among",
    "Throughout", "Furthermore", "Additionally", "Similarly",
    # Adverbs
    "Later", "Even", "Out", "Just", "Then", "Not", "Immediately",
    "Instantly", "Initially", "Halfway", "Midway", "Eventually",
    "Carefully", "Gently", "Graciously", "Excitedly",
    # Participles / gerunds used as sentence openers
    "Wanting", "Excited", "Eager", "Knowing", "Remembering",
    "Realizing", "Understanding", "Following", "Seeing", "Running",
    "Thinking", "Feeling", "Counting", "Delighted", "Enthralled",
    "Intrigued", "Interested", "Impatient", "Adamant", "Busy",
    # Question word / forms of "to be"
    "Is", "Was",
    # Adjectives used as sentence openers
    "However", "Curious",
    # Titles and honorifics
    "Mr", "Mrs", "Ms", "Dr",
    # Days / other nouns that capitalise mid-sentence
    "Saturday", "Sunday", "Friday", "June", "Everyone", "Grandma",
    "Towards", "Arabic", "Other",
    # Last names (spaCy sometimes tags these as PERSON)
    "Smith", "Thompson", "Johnson", "Anderson", "Williams",
}


def extract_names(story: str) -> set[str]:
    """Extract capitalized words mid-sentence that are likely person names."""
    words = re.findall(r"(?<=[a-z ,])([A-Z][a-z]{1,12})(?=[ ,.’’])", story)
    return {w for w in words if w not in SKIP_WORDS}


def build_name_class_index(ds) -> dict[int, str]:
    """Classify each eval example by name type using regex extraction."""
    holdout_set = {c.lower().replace(" ", "_") for c in HOLDOUT_COUNTRIES}
    name_class: dict[int, str] = {}
    for i, ex in enumerate(ds):
        if ex.get("Country", "").lower().replace(" ", "_") in holdout_set:
            continue
        if gold_label(ex) == "neutral":
            continue
        names = extract_names(scenario_text(ex))
        has_non_anglo = bool(names & NON_ANGLO_NAMES)
        has_anglo = bool(names & ANGLO_NAMES)
        if has_non_anglo and has_anglo:
            name_class[i] = "mixed"
        elif has_non_anglo:
            name_class[i] = "non_anglo"
        elif has_anglo:
            name_class[i] = "anglo_only"
        else:
            name_class[i] = "no_name"
    return name_class


def load_normad():
    from datasets import load_dataset
    ds = load_dataset("akhilayerukola/NormAd")
    if hasattr(ds, "keys"):
        for split in ("test", "validation", "train"):
            if split in ds:
                return ds[split]
    return ds


def load_predictions(condition: str, model: str) -> list[dict] | None:
    path = BEHAVIORAL / f"normad_{condition}_{model}_nfs_mpw_usprobe.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def analyze(condition: str, model: str, ds, name_class: dict[int, str]) -> None:
    preds = load_predictions(condition, model)
    if preds is None:
        print(f"  [missing: normad_{condition}_{model}_nfs_mpw_usprobe.json]")
        return

    holdout_set = {c.lower().replace(" ", "_") for c in HOLDOUT_COUNTRIES}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    pred_iter = iter(preds)
    for i, ex in enumerate(ds):
        if ex.get("Country", "").lower().replace(" ", "_") in holdout_set:
            continue
        if gold_label(ex) == "neutral":
            continue
        try:
            p = next(pred_iter)
        except StopIteration:
            break

        cls = name_class.get(i, "no_name")
        correct = p["pred"] == p["gold"]
        us_match = p.get("us_pred") is not None and p["pred"] == p["us_pred"]
        if correct:
            stats[cls]["correct"] += 1
        else:
            stats[cls]["wrong"] += 1
            if us_match:
                stats[cls]["wrong_us"] += 1
        if us_match:
            stats[cls]["us_match"] += 1

    print(f"\n  {condition} / {model}")
    print(f"  {'Name type':<15}  {'N':>6}  {'Acc':>7}  {'US-DR/err':>10}  {'US-DR/all':>10}  {'n_err':>6}")
    print(f"  {'-'*60}")

    for cls in ["anglo_only", "non_anglo", "mixed", "no_name"]:
        s = stats[cls]
        n = s["correct"] + s["wrong"]
        if n == 0:
            continue
        acc = s["correct"] / n
        wrong = s["wrong"]
        dr_err = s["wrong_us"] / wrong if wrong > 0 else float("nan")
        dr_all = s["us_match"] / n
        dr_err_str = f"{dr_err:.1%}" if dr_err == dr_err else "—"
        dr_all_str = f"{dr_all:.1%}"
        print(f"  {cls:<15}  {n:>6}  {acc:.1%}  {dr_err_str:>10}  {dr_all_str:>10}  {wrong:>6}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="8b", choices=["8b", "gemma4"])
    parser.add_argument("--condition", default=None)
    args = parser.parse_args()

    CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult", "sftdpo_aya_nocult"]
    conditions = [args.condition] if args.condition else CONDITIONS

    print("Loading NormAd...")
    ds = load_normad()
    print(f"  {len(ds)} rows")

    name_class = build_name_class_index(ds)

    counts = defaultdict(int)
    for cls in name_class.values():
        counts[cls] += 1
    total = sum(counts.values())
    print(f"\nName type distribution (eval set, {total} examples):")
    print(f"  Anglo only : {counts['anglo_only']:4d}  ({counts['anglo_only']/total:.1%})")
    print(f"  Non-Anglo  : {counts['non_anglo']:4d}  ({counts['non_anglo']/total:.1%})")
    print(f"  Mixed      : {counts['mixed']:4d}  ({counts['mixed']/total:.1%})")
    print(f"  No name    : {counts['no_name']:4d}  ({counts['no_name']/total:.1%})")

    print(f"\n{'='*60}")
    print(f"US default rate by name type  (model={args.model})")
    print(f"{'='*60}")

    for cond in conditions:
        analyze(cond, args.model, ds, name_class)


if __name__ == "__main__":
    main()
