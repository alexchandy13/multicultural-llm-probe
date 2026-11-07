#!/usr/bin/env python3
"""Flag CulturalBench-Easy no-country rewrites that may still name the
culture through a leftover content word, not through the deleted demonym.

Deleting 'Persian' from 'a Persian culture gathering for Nowruz' removes the
explicit marker but leaves 'Nowruz' -- a proper noun that still identifies
Iran. This is a different failure from the grammar bugs fixed earlier: no
rewrite rule catches it, because the leak is in what the question is ABOUT,
not in how it names the country.

This only catches the PROPER-NOUN half of that problem: a capitalized word
surviving away from the sentence start, after the deletion, is a plausible
giveaway (catches 'Nowruz'). It does NOT catch a lowercase giveaway like
'paella' -- that requires recognizing a common noun as a cultural marker,
which has no syntactic signal to key on. Every row this prints needs a human
glance; silence from this script is not a guarantee the row is clean.

  python scripts/flag_cbeasy_leakage.py
  python scripts/flag_cbeasy_leakage.py --show 50
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from evaluate.country_names import strip_country_reference, strip_unregistered_reference

# Capitalized words common enough to appear mid-sentence without being a
# proper-noun giveaway -- question words that got capitalized by a stray
# "Read the following statements" continuation, Roman numerals used as list
# markers, and the like. Short and reviewed by hand rather than guessed at;
# anything not on it gets printed for a human to judge.
_BENIGN = {"I", "II", "III", "IV", "Read", "Yes", "No"}


def suspicious_words(text: str) -> list[str]:
    words = re.findall(r"\b[A-Z][a-zA-Z]*\b", text)
    # Skip the sentence-initial word: its capital is just English orthography.
    first = text.split(maxsplit=1)[0] if text else ""
    return [w for w in words if w != first and w not in _BENIGN]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=30)
    args = ap.parse_args()

    ds = load_dataset("kellycyy/CulturalBench", "CulturalBench-Easy", split="test")

    flagged = []
    for r in ds:
        q, c = r["prompt_question"], r["country"]
        new, applied = strip_country_reference(q, c)
        if not applied:
            new, applied = strip_unregistered_reference(q)
        if not applied:
            continue
        hits = suspicious_words(new)
        if hits:
            flagged.append((c, q, new, hits))

    print(f"{len(flagged)} deleted rows still contain a capitalized word "
          f"other than the first -- check each by eye:")
    for c, old, new, hits in flagged[: args.show]:
        print(f"\n[{c}]  suspicious: {hits}")
        print(f"  before: {old}")
        print(f"  after:  {new}")

    print(f"\nReminder: this list is NOT exhaustive. A lowercase cultural "
          f"marker (a dish, a custom's common name) leaks exactly as much "
          f"and triggers nothing here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
