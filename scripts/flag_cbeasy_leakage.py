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

# A capitalized word is only suspicious if its capital isn't explained by
# position. 'Chewing' in '(i) Chewing loudly' is capitalized because it opens
# a list item, the same reason the text's own first word is capitalized --
# not because it's a proper noun. Checked by looking at everything between
# the start of the word's LINE and the word itself: if that's nothing but
# whitespace and/or a marker like '(i)', '2.', '-', the capital is positional.
_LIST_MARKER = re.compile(
    r"^\s*(?:\(?[ivxlcdm]{1,4}\)|\(?\d{1,2}[.):]?|[-*•])?\s*$", re.I)


def suspicious_words(text: str) -> list[str]:
    hits = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z]*\b", text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start : m.start()]
        if _LIST_MARKER.match(prefix):
            continue
        hits.append(m.group(0))
    return hits


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
