"""Compute country-level Inglehart-Welzel coordinates from raw WVS Wave 7 data,
classify each country into one of 8 cultural clusters, and compute Euclidean
distance from the English-Speaking cluster centroid (our US-anchor proxy).

Input:  data/wvs/WVS_Cross-National_Wave_7_csv_v6_0.csv
Output: data/iw_coordinates.csv

Methodology:
  - WVS Wave 7 pre-computes Welzel's updated I-W indices per respondent:
      SACSECVAL  → Secular Values (modern Traditional ↔ Secular-Rational axis)
      RESEMAVAL  → Emancipative Values (modern Survival ↔ Self-Expression axis)
    Both are 0-1 normalized composites. We aggregate to country level using
    the survey's design weights (W_WEIGHT) — standard survey methodology.

  - Cluster assignment uses the Welzel cultural-zone mapping; for countries
    where a definitive mapping isn't published, we fall back to nearest
    cluster centroid in 2D space.

  - The English-Speaking centroid is computed as the weighted mean of
    SACSECVAL and RESEMAVAL across all English-Speaking countries. Each
    country's distance to this centroid is the headline "cultural distance
    from the US/anglosphere" metric for downstream analysis.

Usage:
    python3.12 analysis/culturemapping/compute_iw_coords.py
    # writes data/iw_coordinates.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WVS_CSV = PROJECT_ROOT / "data" / "wvs" / "WVS_Cross-National_Wave_7_csv_v6_0.csv"
OUT_CSV = PROJECT_ROOT / "data" / "iw_coordinates.csv"

# Only these columns are needed from the 500+ in the raw file. Saves memory.
NEEDED_COLS = {
    "B_COUNTRY",         # ISO 3166-1 numeric country code
    "B_COUNTRY_ALPHA",   # ISO 3166-1 alpha-3 country code
    "W_WEIGHT",          # respondent's survey design weight
    "SACSECVAL",         # Secular Values (0-1)
    "RESEMAVAL",         # Emancipative Values (0-1)
}

# Inglehart-Welzel cultural clusters, based on Welzel's Freedom Rising (2013)
# and the published WVS cultural map. Keyed by ISO 3166-1 alpha-3 code.
# Countries not listed here will be classified by nearest-centroid fallback.
IW_CLUSTERS = {
    # English-Speaking (anglosphere) — our US-anchor cluster
    "USA": "EnglishSpeaking", "GBR": "EnglishSpeaking", "CAN": "EnglishSpeaking",
    "AUS": "EnglishSpeaking", "NZL": "EnglishSpeaking", "IRL": "EnglishSpeaking",

    # Protestant Europe — Northern + Germanic Europe
    "DEU": "ProtestantEurope", "NLD": "ProtestantEurope", "SWE": "ProtestantEurope",
    "DNK": "ProtestantEurope", "NOR": "ProtestantEurope", "FIN": "ProtestantEurope",
    "CHE": "ProtestantEurope", "ISL": "ProtestantEurope", "EST": "ProtestantEurope",

    # Catholic Europe — Southern + Central Europe
    "ITA": "CatholicEurope", "ESP": "CatholicEurope", "FRA": "CatholicEurope",
    "BEL": "CatholicEurope", "PRT": "CatholicEurope", "AUT": "CatholicEurope",
    "LUX": "CatholicEurope", "AND": "CatholicEurope", "MLT": "CatholicEurope",
    "POL": "CatholicEurope", "CZE": "CatholicEurope", "SVK": "CatholicEurope",
    "SVN": "CatholicEurope", "HUN": "CatholicEurope", "HRV": "CatholicEurope",

    # Orthodox Europe / Eurasia
    "RUS": "Orthodox", "UKR": "Orthodox", "BLR": "Orthodox", "BGR": "Orthodox",
    "ROU": "Orthodox", "SRB": "Orthodox", "MKD": "Orthodox", "BIH": "Orthodox",
    "MNE": "Orthodox", "ARM": "Orthodox", "GEO": "Orthodox", "CYP": "Orthodox",
    "MDA": "Orthodox", "GRC": "Orthodox",

    # Confucian / East Asian
    "CHN": "Confucian", "JPN": "Confucian", "KOR": "Confucian", "TWN": "Confucian",
    "HKG": "Confucian", "SGP": "Confucian", "MNG": "Confucian", "MAC": "Confucian",
    "VNM": "Confucian",

    # South Asia / Southeast Asia
    "IND": "SouthAsia", "PAK": "SouthAsia", "BGD": "SouthAsia", "LKA": "SouthAsia",
    "NPL": "SouthAsia", "AFG": "SouthAsia", "IDN": "SouthAsia", "MYS": "SouthAsia",
    "PHL": "SouthAsia", "MMR": "SouthAsia", "THA": "SouthAsia", "KHM": "SouthAsia",
    "LAO": "SouthAsia",

    # African-Islamic — MENA + Sub-Saharan Africa + Muslim-majority Asia
    "TUR": "AfricanIslamic", "AZE": "AfricanIslamic", "IRN": "AfricanIslamic",
    "IRQ": "AfricanIslamic", "JOR": "AfricanIslamic", "LBN": "AfricanIslamic",
    "EGY": "AfricanIslamic", "LBY": "AfricanIslamic", "MAR": "AfricanIslamic",
    "TUN": "AfricanIslamic", "SAU": "AfricanIslamic", "KWT": "AfricanIslamic",
    "QAT": "AfricanIslamic", "ARE": "AfricanIslamic", "DZA": "AfricanIslamic",
    "YEM": "AfricanIslamic", "PSE": "AfricanIslamic", "SYR": "AfricanIslamic",
    "OMN": "AfricanIslamic", "BHR": "AfricanIslamic",
    "NGA": "AfricanIslamic", "KEN": "AfricanIslamic", "ZAF": "AfricanIslamic",
    "ETH": "AfricanIslamic", "GHA": "AfricanIslamic", "ZWE": "AfricanIslamic",
    "MOZ": "AfricanIslamic", "MLI": "AfricanIslamic", "TZA": "AfricanIslamic",
    "RWA": "AfricanIslamic", "UGA": "AfricanIslamic", "BFA": "AfricanIslamic",
    "ISR": "AfricanIslamic",   # debated; Welzel's most recent placements vary

    # Latin America
    "MEX": "LatinAmerica", "BRA": "LatinAmerica", "ARG": "LatinAmerica",
    "CHL": "LatinAmerica", "COL": "LatinAmerica", "PER": "LatinAmerica",
    "VEN": "LatinAmerica", "BOL": "LatinAmerica", "ECU": "LatinAmerica",
    "URY": "LatinAmerica", "PRY": "LatinAmerica", "GTM": "LatinAmerica",
    "NIC": "LatinAmerica", "CRI": "LatinAmerica", "PAN": "LatinAmerica",
    "DOM": "LatinAmerica", "CUB": "LatinAmerica", "JAM": "LatinAmerica",
    "HND": "LatinAmerica", "SLV": "LatinAmerica", "TTO": "LatinAmerica",
    "HTI": "LatinAmerica", "PRI": "LatinAmerica",
}

# ISO alpha-3 → NormAd's lowercase-underscore country name. Only countries that
# appear in BOTH WVS Wave 7 and NormAd are mapped — others are passed through.
NORMAD_NAME = {
    "USA": "united_states_of_america", "GBR": "united_kingdom",
    "CAN": "canada", "AUS": "australia", "NZL": "new_zealand", "IRL": "ireland",
    "DEU": "germany", "NLD": "netherlands", "SWE": "sweden", "AUT": "austria",
    "FRA": "france", "ITA": "italy", "ESP": "spain", "PRT": "portugal",
    "GRC": "greece", "POL": "poland", "CZE": "czech_republic", "HUN": "hungary",
    "ROU": "romania", "HRV": "croatia", "SRB": "serbia", "MKD": "north_macedonia",
    "BIH": "bosnia_and_herzegovina", "CYP": "cyprus", "RUS": "russia", "UKR": "ukraine",
    "CHN": "china", "JPN": "japan", "KOR": "south_korea", "TWN": "taiwan",
    "HKG": "hong_kong", "SGP": "singapore", "VNM": "vietnam",
    "IND": "india", "PAK": "pakistan", "BGD": "bangladesh", "LKA": "sri_lanka",
    "NPL": "nepal", "AFG": "afghanistan", "IDN": "indonesia", "MYS": "malaysia",
    "PHL": "philippines", "THA": "thailand", "MMR": "myanmar", "KHM": "cambodia",
    "LAO": "laos",
    "EGY": "egypt", "IRN": "iran", "IRQ": "iraq", "LBN": "lebanon", "SAU": "saudi_arabia",
    "SYR": "syria", "JOR": "jordan", "TUR": "türkiye", "ISR": "israel",
    "PSE": "palestinian_territories",
    "ETH": "ethiopia", "KEN": "kenya", "ZAF": "south_africa", "NGA": "nigeria",
    "ZWE": "zimbabwe",
    "MEX": "mexico", "BRA": "brazil", "ARG": "argentina", "CHL": "chile",
    "COL": "colombia", "PER": "peru", "VEN": "venezuela",
}


def read_csv_streaming(path: Path):
    """Stream WVS rows one at a time — full file is ~190 MB and we only need 5 columns."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def parse_float(s: str) -> float | None:
    """Return float or None for missing/sentinel values."""
    if s is None or s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    # WVS missing-value sentinels: -1 to -5 (varies), -999, -9999
    if v < -1.5:
        return None
    return v


def aggregate_country_means(csv_path: Path) -> dict[str, dict]:
    """For each country, compute weighted mean SACSECVAL and RESEMAVAL.

    Returns: {alpha3: {"sacsecval": float, "resemaval": float, "n": int}}
    """
    agg = defaultdict(lambda: {
        "sac_num": 0.0, "sac_denom": 0.0,
        "res_num": 0.0, "res_denom": 0.0,
        "n_rows": 0,
    })

    n_total = 0
    for row in read_csv_streaming(csv_path):
        n_total += 1
        if n_total % 20000 == 0:
            print(f"  read {n_total:,} rows...", file=sys.stderr)

        alpha = (row.get("B_COUNTRY_ALPHA") or "").strip().strip('"')
        if not alpha or alpha == "-4":
            continue
        sac = parse_float(row.get("SACSECVAL"))
        res = parse_float(row.get("RESEMAVAL"))
        wt = parse_float(row.get("W_WEIGHT")) or 1.0
        if wt <= 0:
            wt = 1.0

        bucket = agg[alpha]
        bucket["n_rows"] += 1
        if sac is not None:
            bucket["sac_num"] += sac * wt
            bucket["sac_denom"] += wt
        if res is not None:
            bucket["res_num"] += res * wt
            bucket["res_denom"] += wt

    print(f"  total rows read: {n_total:,}", file=sys.stderr)

    out = {}
    for alpha, b in agg.items():
        if b["sac_denom"] == 0 or b["res_denom"] == 0:
            continue
        out[alpha] = {
            "sacsecval": b["sac_num"] / b["sac_denom"],
            "resemaval": b["res_num"] / b["res_denom"],
            "n": b["n_rows"],
        }
    return out


def nearest_centroid(point: tuple[float, float], centroids: dict[str, tuple[float, float]]) -> str:
    """For an unclassified country, assign to the cluster with nearest centroid."""
    best, best_d = None, float("inf")
    for cluster, (cx, cy) in centroids.items():
        d = math.hypot(point[0] - cx, point[1] - cy)
        if d < best_d:
            best_d = d
            best = cluster
    return best


def compute_cluster_centroids(countries: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Mean (sacsecval, resemaval) per cluster, weighted by sample size."""
    by_cluster = defaultdict(lambda: {"sac": 0.0, "res": 0.0, "w": 0.0})
    for alpha, d in countries.items():
        cluster = d.get("cluster")
        if not cluster:
            continue
        n = d["n"]
        by_cluster[cluster]["sac"] += d["sacsecval"] * n
        by_cluster[cluster]["res"] += d["resemaval"] * n
        by_cluster[cluster]["w"] += n
    return {c: (b["sac"] / b["w"], b["res"] / b["w"]) for c, b in by_cluster.items() if b["w"] > 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wvs-csv", default=str(WVS_CSV))
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    csv_path = Path(args.wvs_csv)
    if not csv_path.exists():
        print(f"WVS CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.0f} MB)...", file=sys.stderr)
    countries = aggregate_country_means(csv_path)
    print(f"Aggregated {len(countries)} countries", file=sys.stderr)

    # First pass: assign cluster from published mapping where available.
    for alpha, d in countries.items():
        d["cluster"] = IW_CLUSTERS.get(alpha)

    # Compute centroids from mapped countries.
    centroids = compute_cluster_centroids(countries)
    print(f"\nCluster centroids (sacsecval, resemaval):", file=sys.stderr)
    for c, (sx, sy) in sorted(centroids.items()):
        print(f"  {c:18s}  ({sx:.3f}, {sy:.3f})", file=sys.stderr)

    # Second pass: assign unmapped countries via nearest-centroid.
    for alpha, d in countries.items():
        if d.get("cluster") is None:
            d["cluster"] = nearest_centroid((d["sacsecval"], d["resemaval"]), centroids)
            d["cluster_inferred"] = True
        else:
            d["cluster_inferred"] = False

    # Anchor: English-Speaking centroid.
    eng_centroid = centroids.get("EnglishSpeaking")
    if eng_centroid is None:
        print("WARNING: No English-Speaking countries in data; using USA alone as anchor", file=sys.stderr)
        if "USA" in countries:
            eng_centroid = (countries["USA"]["sacsecval"], countries["USA"]["resemaval"])
        else:
            print("ERROR: No USA data either; cannot compute distances", file=sys.stderr)
            sys.exit(2)

    # Compute per-country distance to English-Speaking centroid.
    for alpha, d in countries.items():
        d["dist_from_english"] = math.hypot(
            d["sacsecval"] - eng_centroid[0],
            d["resemaval"] - eng_centroid[1],
        )

    # === Augmentation pass ===========================================
    # WVS Wave 7 only surveyed ~60 countries; many NormAd countries with known
    # I-W cluster assignments (Ireland, Sweden, France, Italy, Spain, Portugal,
    # Saudi Arabia, Afghanistan, …) are absent. We emit rows for every
    # (ISO, cluster) entry in IW_CLUSTERS that has a NORMAD_NAME mapping but
    # was missing from WVS, with empty coord fields. Downstream scripts that
    # only need cluster membership (e.g. analysis/significance_tests.py) work
    # immediately; scripts that need coords filter on non-empty.
    augmented = 0
    for alpha, cluster in IW_CLUSTERS.items():
        if alpha in countries:
            continue
        if alpha not in NORMAD_NAME:
            continue
        countries[alpha] = {
            "cluster": cluster,
            "cluster_inferred": False,
            "sacsecval": None,
            "resemaval": None,
            "dist_from_english": None,
            "n": 0,
            "_augmented": True,
        }
        augmented += 1
    print(f"\nAugmented with {augmented} NormAd countries missing from WVS Wave 7 "
          f"(empty coords, cluster from published I-W map).", file=sys.stderr)

    # Write output CSV, sorted by distance ascending (augmented rows at the end).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        countries.items(),
        key=lambda kv: (kv[1].get("dist_from_english") is None,
                        kv[1].get("dist_from_english") or 0.0),
    )
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country_iso", "normad_country", "cluster", "cluster_inferred",
                    "sacsecval", "resemaval", "dist_from_english", "n_respondents"])
        for alpha, d in rows:
            def fmt(v):
                return f"{v:.4f}" if v is not None else ""
            w.writerow([
                alpha,
                NORMAD_NAME.get(alpha, ""),
                d["cluster"],
                "yes" if d.get("cluster_inferred") else "no",
                fmt(d["sacsecval"]),
                fmt(d["resemaval"]),
                fmt(d["dist_from_english"]),
                d["n"],
            ])
    print(f"\nWrote {out_path} ({len(rows)} countries; {augmented} augmented)",
          file=sys.stderr)

    # Print per-cluster summary to stderr. Filter to countries with non-null
    # distance (i.e., exclude augmented rows that lack WVS coords).
    print(f"\nCluster summary (mean dist from EnglishSpeaking centroid):", file=sys.stderr)
    by_cluster = defaultdict(list)
    for alpha, d in countries.items():
        if d.get("dist_from_english") is not None:
            by_cluster[d["cluster"]].append(d["dist_from_english"])
    for cluster in sorted(by_cluster, key=lambda c: sum(by_cluster[c]) / len(by_cluster[c])):
        ds = by_cluster[cluster]
        n_in_cluster = sum(1 for a, dd in countries.items() if dd["cluster"] == cluster)
        print(f"  {cluster:18s}  mean dist = {sum(ds)/len(ds):.3f}  "
              f"(n={n_in_cluster} countries; {len(ds)} with coords)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
