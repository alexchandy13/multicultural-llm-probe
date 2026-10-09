"""Map a score-file country key to its Inglehart-Welzel cultural cluster.

CULNIG score files key per-country attributions by whatever spelling the source
benchmark used, and the three benchmarks disagree: NormAd writes
"united_states_of_america" (or "USA", once rev_c2n has mapped it), CulturalBench
writes "United States", BLEnD writes "US". BLEnD also ships sub-national regions
("Assam", "West_Java", "Northern_Nigeria") that have no ISO code of their own.

The cluster assignment itself is not redefined here — it is imported from
analysis.compute_iw_coords, which holds IW_CLUSTERS (ISO alpha-3 -> cluster) and
is the same map the accuracy analyses use. This module only resolves names to
alpha-3 codes so the two can be joined.

cluster_of() returns None for anything unresolvable rather than guessing, and
report_coverage() exists so a selection run can print what it dropped instead of
silently excluding countries from a cluster mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.compute_iw_coords import IW_CLUSTERS, NORMAD_NAME, NORMAD_ONLY

CLUSTERS = [
    "EnglishSpeaking", "ProtestantEurope", "CatholicEurope", "Orthodox",
    "Confucian", "SouthAsia", "AfricanIslamic", "LatinAmerica",
]

# alpha-3 -> cluster. NORMAD_ONLY covers countries absent from every WVS wave,
# which IW_CLUSTERS therefore does not place.
ALPHA_TO_CLUSTER: dict[str, str] = {**NORMAD_ONLY, **IW_CLUSTERS}


def _norm(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip().lower()


# Spellings NORMAD_NAME does not carry, plus the sub-national BLEnD regions,
# which are folded into the country that contains them.
_EXTRA_ALIASES: dict[str, str] = {
    # USA / UK variants across the three benchmarks
    "usa": "USA", "us": "USA", "united states": "USA",
    "united states of america": "USA",
    "uk": "GBR", "united kingdom": "GBR", "great britain": "GBR",
    # endonym/exonym pairs
    "turkey": "TUR", "türkiye": "TUR",
    "czechia": "CZE", "czech republic": "CZE",
    # BLEnD-only countries
    "algeria": "DZA", "azerbaijan": "AZE", "north korea": "PRK",
    # BLEnD sub-national regions -> containing country
    "assam": "IND", "west java": "IDN", "northern nigeria": "NGA",
    # CulturalBench-only countries
    "morocco": "MAR",
}

# Countries the above introduces that IW_CLUSTERS does not place. Assigned on the
# same basis compute_iw_coords uses for NORMAD_ONLY: nearest cultural neighbour
# with a published WVS position.
_EXTRA_CLUSTERS: dict[str, str] = {
    "DZA": "AfricanIslamic",
    "MAR": "AfricanIslamic",
    "AZE": "Orthodox",      # post-Soviet, groups with Orthodox on the WVS map
    "PRK": "Confucian",
}

NAME_TO_ALPHA: dict[str, str] = {_norm(v): k for k, v in NORMAD_NAME.items()}
NAME_TO_ALPHA.update(_EXTRA_ALIASES)
_CLUSTER_OF_ALPHA: dict[str, str] = {**_EXTRA_CLUSTERS, **ALPHA_TO_CLUSTER}


def alpha_of(country: str) -> str | None:
    return NAME_TO_ALPHA.get(_norm(country))


def cluster_of(country: str) -> str | None:
    """Return the IW cluster for a score-file country key, or None if unknown."""
    alpha = alpha_of(country)
    return _CLUSTER_OF_ALPHA.get(alpha) if alpha else None


def cluster_index() -> dict[str, int]:
    return {c: i for i, c in enumerate(CLUSTERS)}


def report_coverage(countries) -> tuple[dict[str, list[str]], list[str]]:
    """Split countries into {cluster: [countries]} and a list of unresolved ones."""
    by_cluster: dict[str, list[str]] = {c: [] for c in CLUSTERS}
    unresolved: list[str] = []
    for c in countries:
        cl = cluster_of(c)
        if cl is None:
            unresolved.append(c)
        else:
            by_cluster[cl].append(c)
    return by_cluster, unresolved
