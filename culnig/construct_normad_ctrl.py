"""Regex utilities for stripping cultural context from NormAd story text.

The plan's Step 2 says: "rule-based regex to strip country/culture references from
scenario prefix and body". This module is imported by `culnig.dataset_ext` to apply
that stripping when constructing the `normadcontrol` dataset block in-flight.

Standalone usage just produces a manual-verification dump so a human can spot-check
that the stripping removes culture markers without breaking scenario grammar:

    python culnig/construct_normad_ctrl.py --verify-sample 30
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from datasets import load_dataset


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
    return "|".join(re.escape(x) for x in sorted(set(items), key=len, reverse=True))


COUNTRY_ALT = _alt(COUNTRIES)
DEMONYM_ALT = _alt(DEMONYMS)

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
    """Strip country/demonym mentions from a NormAd story; preserve structure."""
    out = text or ""
    for pat in PREFIX_PATTERNS:
        out = pat.sub("", out, count=1)
    for pat in INLINE_PATTERNS:
        out = pat.sub("", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-sample", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-path",
        default=str(PROJECT_ROOT / "data" / "NormAdctrl" / "manual_verify.txt"),
    )
    args = parser.parse_args()

    ds = load_dataset("akhilayerukola/NormAd", split="train")
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(ds)), min(args.verify_sample, len(ds))))

    lines = [
        "# Manual verification — NormAd story vs. stripped NormAdctrl story",
        "# Confirm culture markers removed; scenario grammar preserved.\n",
    ]
    for i in idxs:
        ex = ds[i]
        orig = ex.get("Story", "")
        stripped = strip_culture(orig)
        lines.append(f"--- idx={i} country={ex.get('Country')} ---")
        lines.append(f"BEFORE: {orig}")
        lines.append(f"AFTER : {stripped}\n")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
