"""Language breakdown of training splits by I-W cluster.

For each of the 4 splits (aya_cult, aya_nocult, dpo_cult, dpo_nocult), counts
examples per language, then maps languages to I-W clusters. Languages widely
spoken across multiple clusters (e.g. French, Spanish) are counted in each
applicable cluster — so cluster totals can exceed the split total.

Usage (run from repo root on cluster):
    python analysis/training_language_clusters.py
    python analysis/training_language_clusters.py --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# language name (as it appears in Aya dataset) → list of I-W clusters
# Multi-cluster = language is a major language in countries from multiple clusters.
LANG_TO_CLUSTERS: dict[str, list[str]] = {
    # EnglishSpeaking
    "English":                   ["EnglishSpeaking"],

    # ProtestantEurope
    "German":                    ["ProtestantEurope", "CatholicEurope"],  # Germany + Austria
    "Dutch":                     ["ProtestantEurope"],
    "Swedish":                   ["ProtestantEurope"],
    "Norwegian":                 ["ProtestantEurope"],
    "Danish":                    ["ProtestantEurope"],
    "Finnish":                   ["ProtestantEurope"],
    "Estonian":                  ["ProtestantEurope"],
    "Latvian":                   ["ProtestantEurope"],
    "Lithuanian":                ["ProtestantEurope"],

    # CatholicEurope
    "Italian":                   ["CatholicEurope"],
    "Polish":                    ["CatholicEurope"],
    "Czech":                     ["CatholicEurope"],
    "Slovak":                    ["CatholicEurope"],
    "Hungarian":                 ["CatholicEurope"],
    "Croatian":                  ["CatholicEurope"],
    "Slovenian":                 ["CatholicEurope"],
    "Catalan":                   ["CatholicEurope"],
    "Basque":                    ["CatholicEurope"],
    "Galician":                  ["CatholicEurope"],

    # Multi-cluster: CatholicEurope + LatinAmerica
    "Spanish":                   ["CatholicEurope", "LatinAmerica"],
    "Portuguese":                ["CatholicEurope", "LatinAmerica"],
    # Multi-cluster: CatholicEurope + AfricanIslamic (large francophone Africa)
    "French":                    ["CatholicEurope", "AfricanIslamic"],

    # LatinAmerica — no major language exclusive to this cluster beyond Spanish/Portuguese

    # Confucian
    "Chinese Simplified":        ["Confucian"],
    "Chinese Traditional":       ["Confucian"],
    "Chinese (Simplified)":      ["Confucian"],
    "Chinese (Traditional)":     ["Confucian"],
    "Japanese":                  ["Confucian"],
    "Korean":                    ["Confucian"],
    "Vietnamese":                ["Confucian"],

    # Orthodox
    "Russian":                   ["Orthodox"],
    "Ukrainian":                 ["Orthodox"],
    "Bulgarian":                 ["Orthodox"],
    "Serbian":                   ["Orthodox"],
    "Greek":                     ["Orthodox"],
    "Romanian":                  ["Orthodox", "CatholicEurope"],  # straddles
    "Georgian":                  ["Orthodox"],
    "Armenian":                  ["Orthodox"],
    "Belarusian":                ["Orthodox"],

    # SouthAsia
    "Hindi":                     ["SouthAsia"],
    "Bengali":                   ["SouthAsia"],
    "Urdu":                      ["SouthAsia", "AfricanIslamic"],  # Pakistan + wide Islamic use
    "Tamil":                     ["SouthAsia"],
    "Telugu":                    ["SouthAsia"],
    "Kannada":                   ["SouthAsia"],
    "Malayalam":                 ["SouthAsia"],
    "Marathi":                   ["SouthAsia"],
    "Punjabi":                   ["SouthAsia"],
    "Gujarati":                  ["SouthAsia"],
    "Odia":                      ["SouthAsia"],
    "Oriya":                     ["SouthAsia"],
    "Nepali":                    ["SouthAsia"],
    "Sinhala":                   ["SouthAsia"],
    "Assamese":                  ["SouthAsia"],
    "Konkani":                   ["SouthAsia"],

    # AfricanIslamic
    "Arabic":                    ["AfricanIslamic"],
    "Swahili":                   ["AfricanIslamic"],
    "Hausa":                     ["AfricanIslamic"],
    "Yoruba":                    ["AfricanIslamic"],
    "Igbo":                      ["AfricanIslamic"],
    "Amharic":                   ["AfricanIslamic"],
    "Somali":                    ["AfricanIslamic"],
    "Wolof":                     ["AfricanIslamic"],
    "Zulu":                      ["AfricanIslamic"],
    "Xhosa":                     ["AfricanIslamic"],
    "Shona":                     ["AfricanIslamic"],
    "Kinyarwanda":               ["AfricanIslamic"],
    "Lingala":                   ["AfricanIslamic"],
    "Luganda":                   ["AfricanIslamic"],
    "Tigrinya":                  ["AfricanIslamic"],
    "Turkish":                   ["AfricanIslamic"],  # straddles Orthodox but culturally Islamic
    "Persian":                   ["AfricanIslamic"],
    "Farsi":                     ["AfricanIslamic"],
    "Kurdish":                   ["AfricanIslamic"],
    "Malay":                     ["AfricanIslamic"],
    "Indonesian":                ["AfricanIslamic"],
    "Uzbek":                     ["AfricanIslamic"],
    "Kazakh":                    ["AfricanIslamic"],
    "Azerbaijani":               ["AfricanIslamic"],
    "Pashto":                    ["SouthAsia", "AfricanIslamic"],
    "Sindhi":                    ["SouthAsia", "AfricanIslamic"],

    # Additional languages
    "Panjabi":                   ["SouthAsia"],   # variant spelling of Punjabi
    "Malagasy":                  ["AfricanIslamic"],
    "Kyrgyz":                    ["AfricanIslamic"],
    "Tagalog":                   ["LatinAmerica"],  # Philippines — shared Spanish Catholic colonial heritage
    "Cebuano":                   ["LatinAmerica"],  # Philippines
    "Chinese":                   ["Confucian"],   # unspecified script variant
    "Thai":                      ["Confucian"],   # Buddhist Southeast Asia
    "Khmer":                     ["Confucian"],   # Cambodia
    "Hebrew":                    ["ProtestantEurope"],  # Israel clusters near secular-rational/self-expression on IW map
    "Macedonian":                ["Orthodox"],
    "Welsh":                     ["EnglishSpeaking"],
    "Irish":                     ["EnglishSpeaking"],
    "Waray":                     ["LatinAmerica"],       # Philippines
    "Sanskrit":                  ["SouthAsia"],
    "Sundanese":                 ["AfricanIslamic"],     # West Java, predominantly Muslim
    "Burmese":                   ["Confucian"],          # Myanmar, Buddhist Southeast Asia
    "Lao":                       ["Confucian"],          # Laos, Buddhist Southeast Asia
    "Yiddish":                   ["ProtestantEurope"],   # European Jewish diaspora
    "Latin":                     ["CatholicEurope"],
    "Esperanto":                 ["EnglishSpeaking"],    # no geographic cluster; speaker base mostly Western

    # Round 3
    "Javanese":                  ["AfricanIslamic"],     # Java, Indonesia — Muslim majority
    "Albanian":                  ["AfricanIslamic"],     # Albania — Muslim majority
    "Tajik":                     ["AfricanIslamic"],     # Tajikistan — Central Asian Muslim
    "Lombard":                   ["CatholicEurope"],     # Northern Italy/Switzerland
    "Tatar":                     ["AfricanIslamic"],     # Turkic Muslim people of Russia
    "Iloko":                     ["LatinAmerica"],       # Philippines
    "Aragonese":                 ["CatholicEurope"],     # Spain
    "Haitian Creole":            ["LatinAmerica"],       # Haiti
    "Bashkir":                   ["AfricanIslamic"],     # Turkic Muslim, Russia
    "Limburgish":                ["ProtestantEurope"],   # Netherlands/Belgium border
    "Sicilian":                  ["CatholicEurope"],
    "Sardinian":                 ["CatholicEurope"],
    "Afrikaans":                 ["AfricanIslamic"],     # South Africa
    "Uyghur":                    ["AfricanIslamic"],     # Xinjiang — Turkic Muslim
    "Asturian":                  ["CatholicEurope"],     # Spain
    "Mongolian":                 ["Confucian"],          # Mongolia
    "Icelandic":                 ["ProtestantEurope"],
    "Luxembourgish":             ["CatholicEurope"],
    "Goan Konkani":              ["SouthAsia"],          # Goa, India
    "Maltese":                   ["CatholicEurope"],
    "Bosnian":                   ["AfricanIslamic"],     # Bosnia — Muslim majority
    "Occitan":                   ["CatholicEurope"],     # Southern France/Spain
    "Minangkabau":               ["AfricanIslamic"],     # West Sumatra, Indonesia — Muslim
    "Venetian":                  ["CatholicEurope"],     # Northern Italy
    "Guarani":                   ["LatinAmerica"],       # Paraguay
    "Turkmen":                   ["AfricanIslamic"],     # Central Asia — Muslim
    "Scottish Gaelic":           ["EnglishSpeaking"],    # Scotland
    "Maithili":                  ["SouthAsia"],          # Bihar, India/Nepal
    "Eastern Mari":              ["Orthodox"],           # Finno-Ugric, Russia
    "Tuvan":                     ["Orthodox"],           # Russia/Mongolia border
}

IW_ORDER = [
    "EnglishSpeaking", "ProtestantEurope", "CatholicEurope", "Confucian",
    "LatinAmerica", "Orthodox", "SouthAsia", "AfricanIslamic",
]

SPLITS = ["aya_cult", "aya_nocult", "dpo_cult", "dpo_nocult"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    col_w = 18
    hdr = f"{'split':<18}" + "".join(f"{c:<{col_w}}" for c in IW_ORDER) + f"{'unmapped':<12}{'total'}"
    print(hdr)
    print("-" * len(hdr))

    grand_cluster: dict[str, int] = defaultdict(int)
    grand_unmapped = 0
    grand_total = 0

    for split in SPLITS:
        path = data_dir / split
        if not path.exists():
            print(f"{split:<18}  [not found]")
            continue
        ds = load_from_disk(str(path))
        lang_counts = Counter(ds["language"])
        total = len(ds)

        cluster_counts: dict[str, int] = defaultdict(int)
        unmapped_count = 0
        unmapped_langs: Counter = Counter()

        for lang, n in lang_counts.items():
            clusters = LANG_TO_CLUSTERS.get(lang)
            if clusters:
                for c in clusters:
                    cluster_counts[c] += n
                    grand_cluster[c] += n
            else:
                unmapped_count += n
                unmapped_langs[lang] += n

        grand_unmapped += unmapped_count
        grand_total += total

        row = f"{split:<18}" + "".join(
            f"{100*cluster_counts.get(c,0)/total:>{col_w-2}.1f}% " for c in IW_ORDER
        ) + f"{100*unmapped_count/total:>9.1f}%  {total}"
        print(row)

        if unmapped_langs:
            print(f"  {'':16}  unmapped langs: " +
                  ", ".join(f"{l}={n}" for l, n in unmapped_langs.most_common()))

    print("-" * len(hdr))
    total_row = f"{'TOTAL':<18}" + "".join(
        f"{100*grand_cluster.get(c,0)/grand_total:>{col_w-2}.1f}% " for c in IW_ORDER
    ) + f"{100*grand_unmapped/grand_total:>9.1f}%  {grand_total}"
    print(total_row)

    print()
    print("Note: multi-cluster languages (French, Spanish, Portuguese, etc.) are")
    print("counted in each applicable cluster, so cluster totals can exceed split total.")

    # Also print per-split language breakdown
    print()
    print("=== Top 15 languages per split ===")
    for split in SPLITS:
        path = data_dir / split
        if not path.exists():
            continue
        ds = load_from_disk(str(path))
        counts = Counter(ds["language"]).most_common(15)
        total = len(ds)
        print(f"\n{split} (n={total:,}):")
        for lang, n in counts:
            clusters = LANG_TO_CLUSTERS.get(lang, ["?"])
            print(f"  {lang:<35} {n:>6}  ({100*n/total:.1f}%)  → {', '.join(clusters)}")


if __name__ == "__main__":
    main()
