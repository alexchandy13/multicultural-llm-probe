"""Construct NormAdctrl by stripping cultural context from NormAd scenarios.

NormAd: "In Japan, is it acceptable to pour your own drink at a dinner party?"
NormAdctrl: "Is it acceptable to pour your own drink at a dinner party?"

Run pattern (per plan §"Step 2"):
  1. Rule-based regex strips country/culture references from scenario text.
  2. Sample 20-30 examples for manual verification — written to manual_verify.txt.
  3. Final stripped set saved to data/NormAdctrl/ in the same schema as NormAd.

Usage:
    python culnig/construct_normad_ctrl.py
    python culnig/construct_normad_ctrl.py --verify-only  # just regenerate the sample
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COUNTRIES = [
    "Afghanistan", "Algeria", "Argentina", "Australia", "Austria", "Bangladesh",
    "Belgium", "Brazil", "Canada", "Chile", "China", "Colombia", "Czech Republic",
    "Denmark", "Egypt", "Ethiopia", "Finland", "France", "Germany", "Ghana",
    "Greece", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
    "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Korea", "Lebanon",
    "Malaysia", "Mexico", "Morocco", "Netherlands", "New Zealand", "Nigeria",
    "Norway", "Pakistan", "Peru", "Philippines", "Poland", "Portugal", "Romania",
    "Russia", "Saudi Arabia", "Singapore", "South Africa", "South Korea", "Spain",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Thailand", "Tunisia", "Turkey",
    "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "UK",
    "USA", "US", "America", "Vietnam", "Yemen", "Zimbabwe",
]
DEMONYMS = [
    "American", "British", "Chinese", "Indian", "Japanese", "Korean", "Mexican",
    "Iranian", "Indonesian", "Nigerian", "German", "Spanish", "French", "Italian",
    "Russian", "Brazilian", "Turkish", "Egyptian", "Saudi", "Pakistani",
    "Bangladeshi", "Vietnamese", "Thai", "Filipino", "Malaysian", "Dutch", "Polish",
    "Greek", "Israeli", "Australian", "Canadian", "Swiss", "Swedish", "Norwegian",
    "Danish", "Finnish", "Argentinian", "Colombian", "Chilean", "Peruvian",
]


def _alt(items: list[str]) -> str:
    items = sorted(set(items), key=len, reverse=True)  # longest first
    return "|".join(re.escape(x) for x in items)


COUNTRY_ALT = _alt(COUNTRIES)
DEMONYM_ALT = _alt(DEMONYMS)

# "In <Country>, ..."  / "In a <Demonym> ..."
PREFIX_PATTERNS = [
    re.compile(rf"^\s*In\s+(?:the\s+)?(?:{COUNTRY_ALT})\s*,\s*", re.IGNORECASE),
    re.compile(rf"^\s*In\s+(?:a|an|the)?\s*(?:{DEMONYM_ALT})\s+", re.IGNORECASE),
    re.compile(rf"^\s*Among\s+(?:{DEMONYM_ALT})s?\s*,\s*", re.IGNORECASE),
]
INLINE_PATTERNS = [
    re.compile(rf"\bin\s+(?:the\s+)?(?:{COUNTRY_ALT})\b", re.IGNORECASE),
    re.compile(rf"\b(?:{DEMONYM_ALT})\b", re.IGNORECASE),
    re.compile(rf"\bfrom\s+(?:{COUNTRY_ALT})\b", re.IGNORECASE),
]


def strip_culture(text: str) -> str:
    out = text or ""
    for pat in PREFIX_PATTERNS:
        out = pat.sub("", out, count=1)
    for pat in INLINE_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def transform_example(ex: dict) -> dict:
    new = dict(ex)
    for key in ("story", "scenario", "situation", "text"):
        if key in new and isinstance(new[key], str):
            new[key] = strip_culture(new[key])
    new["country"] = "UNSPECIFIED"
    new["_orig_country"] = ex.get("country") or ex.get("Country")
    return new


def load_normad(path: Path):
    if path.exists() and any(path.iterdir()):
        return load_from_disk(str(path))
    ds = load_dataset("akhilayerukola/NormAd")
    return ds["test"] if "test" in ds else ds["train"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normad-path", default=str(PROJECT_ROOT / "data" / "NormAd"))
    parser.add_argument("--out-path", default=str(PROJECT_ROOT / "data" / "NormAdctrl"))
    parser.add_argument("--verify-sample", type=int, default=25)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_normad(Path(args.normad_path))
    transformed = ds.map(transform_example)

    out = Path(args.out_path)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n = len(transformed)
    sample_idxs = sorted(rng.sample(range(n), min(args.verify_sample, n)))
    verify_lines = ["# Manual verification — NormAd vs. NormAdctrl"]
    verify_lines.append("# Confirm cultural specificity removed; scenario structure preserved.\n")
    for i in sample_idxs:
        orig = ds[i]
        stripped = transformed[i]
        orig_text = orig.get("story") or orig.get("scenario") or orig.get("situation") or ""
        new_text = stripped.get("story") or stripped.get("scenario") or stripped.get("situation") or ""
        verify_lines.append(f"--- idx={i} country={orig.get('country')} ---")
        verify_lines.append(f"BEFORE: {orig_text}")
        verify_lines.append(f"AFTER : {new_text}\n")
    (out / "manual_verify.txt").write_text("\n".join(verify_lines))
    print(f"Wrote manual sample to {out/'manual_verify.txt'}")

    if args.verify_only:
        return

    transformed.save_to_disk(str(out))
    print(f"Wrote NormAdctrl ({len(transformed)} examples) to {out}")


if __name__ == "__main__":
    main()
