"""Compute country-level Inglehart-Welzel coordinates from WVS waves 4–7,
classify each country into one of 8 cultural clusters, and compute Euclidean
distance from the English-Speaking cluster centroid (our US-anchor proxy).

Inputs (data/wvs/, most-recent-wins per country):
  WVS_Cross-National_Wave_7_csv_v6_0.csv  (comma-delimited, B_COUNTRY_ALPHA)
  WV6_Data_csv_v20201117.csv              (semicolon-delimited, B_COUNTRY_ALPHA)
  WV5_Data_csv_v20180912.csv              (semicolon-delimited, numeric V2A)
  WV4_Data_csv_v20201117.csv              (semicolon-delimited, B_COUNTRY_ALPHA)
Output: data/iw_coordinates.csv

Methodology:
  - Each WVS wave pre-computes Welzel's I-W indices per respondent:
      SACSECVAL  → Secular Values (Traditional ↔ Secular-Rational axis)
      RESEMAVAL  → Emancipative Values (Survival ↔ Self-Expression axis)
    Both are 0-1 normalized. We aggregate to country level using design
    weights (W_WEIGHT where available, else unweighted).
  - For each country, we use the most recent wave that has data.
  - Cluster assignment uses the Welzel cultural-zone mapping; unclassified
    countries fall back to nearest cluster centroid.

Usage:
    python analysis/compute_iw_coords.py
    # writes data/iw_coordinates.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WVS_DIR = PROJECT_ROOT / "data" / "wvs"
OUT_CSV  = PROJECT_ROOT / "data" / "iw_coordinates.csv"

# WVS wave files in descending priority (most recent first).
# Each entry: (filename, delimiter, alpha_col, numeric_col)
# numeric_col is used when alpha_col is absent (WV5).
WVS_WAVES = [
    ("WVS_Cross-National_Wave_7_csv_v6_0.csv", ",",  "B_COUNTRY_ALPHA", None),
    ("WV6_Data_csv_v20201117.csv",              ";",  "B_COUNTRY_ALPHA", None),
    ("WV5_Data_csv_v20180912.csv",              ";",  None,              "V2A"),
    ("WV4_Data_csv_v20201117.csv",              ";",  "B_COUNTRY_ALPHA", None),
]

# WVS Wave 5 numeric country code → ISO alpha-3
WV5_CODE_TO_ISO = {
    20:"AND", 31:"ARG", 36:"AUS", 40:"AUT", 50:"BGD", 76:"BRA", 100:"BGR",
    116:"KHM", 124:"CAN", 144:"LKA", 152:"CHL", 156:"CHN", 170:"COL",
    191:"HRV", 196:"CYP", 203:"CZE", 208:"DNK", 218:"ECU", 231:"ETH",
    246:"FIN", 250:"FRA", 268:"GEO", 276:"DEU", 288:"GHA", 300:"GRC",
    344:"HKG", 348:"HUN", 356:"IND", 360:"IDN", 364:"IRN", 368:"IRQ",
    372:"IRL", 376:"ISR", 380:"ITA", 388:"JAM", 392:"JPN", 398:"KAZ",
    400:"JOR", 404:"KEN", 410:"KOR", 418:"LAO", 422:"LBN", 458:"MYS",
    484:"MEX", 504:"MAR", 524:"NPL", 528:"NLD", 554:"NZL", 566:"NGA",
    578:"NOR", 586:"PAK", 604:"PER", 608:"PHL", 616:"POL", 620:"PRT",
    630:"PRI", 642:"ROU", 643:"RUS", 682:"SAU", 688:"SRB", 703:"SVK",
    705:"SVN", 710:"ZAF", 724:"ESP", 752:"SWE", 756:"CHE", 760:"SYR",
    764:"THA", 788:"TUN", 792:"TUR", 804:"UKR", 807:"MKD", 826:"GBR",
    840:"USA", 858:"URY", 862:"VEN", 704:"VNM", 716:"ZWE",  70:"BIH",
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

# ISO alpha-3 → NormAd's lowercase-underscore country name.
NORMAD_NAME = {
    "USA": "united_states_of_america", "GBR": "uk",
    "CAN": "canada", "AUS": "australia", "NZL": "new_zealand", "IRL": "ireland",
    "DEU": "germany", "NLD": "netherlands", "SWE": "sweden", "AUT": "austria",
    "FRA": "france", "ITA": "italy", "ESP": "spain", "PRT": "portugal",
    "GRC": "greece", "POL": "poland", "CZE": "czech_republic", "HUN": "hungary",
    "ROU": "romania", "HRV": "croatia", "SRB": "serbia", "MKD": "north_macedonia",
    "BIH": "bosnia_and_herzegovina", "CYP": "cyprus", "RUS": "russia", "UKR": "ukraine",
    "MLT": "malta",
    "CHN": "china", "JPN": "japan", "KOR": "south_korea", "TWN": "taiwan",
    "HKG": "hong_kong", "SGP": "singapore", "VNM": "vietnam",
    "IND": "india", "PAK": "pakistan", "BGD": "bangladesh", "LKA": "sri_lanka",
    "NPL": "nepal", "AFG": "afghanistan", "IDN": "indonesia", "MYS": "malaysia",
    "PHL": "philippines", "THA": "thailand", "MMR": "myanmar", "KHM": "cambodia",
    "LAO": "laos", "TLS": "timor-leste",
    "EGY": "egypt", "IRN": "iran", "IRQ": "iraq", "LBN": "lebanon", "SAU": "saudi_arabia",
    "SYR": "syria", "JOR": "jordan", "TUR": "türkiye", "ISR": "israel",
    "PSE": "palestinian_territories", "SOM": "somalia",
    "ETH": "ethiopia", "KEN": "kenya", "ZAF": "south_africa", "NGA": "nigeria",
    "ZWE": "zimbabwe", "SSD": "south_sudan", "SDN": "sudan",
    "MEX": "mexico", "BRA": "brazil", "ARG": "argentina", "CHL": "chile",
    "COL": "colombia", "PER": "peru", "VEN": "venezuela",
    # Pacific — not in WVS; cluster centroid used as fallback
    "FJI": "fiji", "WSM": "samoa", "TON": "tonga", "PNG": "papua_new_guinea",
    "MUS": "mauritius",
}

# NormAd countries not in any WVS wave — assign cluster here; coordinates
# will be filled from cluster centroid after centroids are computed.
NORMAD_ONLY = {
    "AUT": "CatholicEurope",   # not in any wave with valid scores
    "HRV": "CatholicEurope",
    "IRL": "EnglishSpeaking",
    "PRT": "CatholicEurope",
    "ISR": "AfricanIslamic",
    "SYR": "AfricanIslamic",
    "LKA": "SouthAsia",
    "NPL": "SouthAsia",
    "AFG": "SouthAsia",
    "KHM": "SouthAsia",
    "LAO": "SouthAsia",
    "FJI": "SouthAsia",        # Melanesia — nearest IW cluster
    "WSM": "SouthAsia",
    "TON": "SouthAsia",
    "PNG": "SouthAsia",
    "TLS": "SouthAsia",
    "MUS": "AfricanIslamic",
    "MLT": "CatholicEurope",
    "SOM": "AfricanIslamic",
    "SSD": "AfricanIslamic",
    "SDN": "AfricanIslamic",
}


def parse_float(s: str) -> float | None:
    """Return float or None for missing/sentinel values."""
    if s is None or s == "":
        return None
    try:
        v = float(str(s).replace(",", "."))
    except ValueError:
        return None
    if v < -1.5:
        return None
    return v


def aggregate_wave(path: Path, delimiter: str, alpha_col: str | None,
                   numeric_col: str | None) -> dict[str, dict]:
    """Aggregate one WVS wave file to country-level means.

    Returns {iso_alpha3: {"sacsecval": float, "resemaval": float, "n": int}}
    """
    agg: dict = defaultdict(lambda: {
        "sac_num": 0.0, "sac_denom": 0.0,
        "res_num": 0.0, "res_denom": 0.0,
        "n_rows": 0,
    })
    encoding = "utf-8-sig"
    with open(path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if alpha_col:
                alpha = (row.get(alpha_col) or "").strip().strip('"')
            else:
                try:
                    code = int(float((row.get(numeric_col) or "0").replace(",",".")))
                    alpha = WV5_CODE_TO_ISO.get(code, "")
                except (ValueError, TypeError):
                    alpha = ""
            if not alpha or alpha == "-4":
                continue
            sac = parse_float(row.get("SACSECVAL"))
            res = parse_float(row.get("RESEMAVAL"))
            wt  = parse_float(row.get("W_WEIGHT")) or 1.0
            if wt <= 0:
                wt = 1.0
            b = agg[alpha]
            b["n_rows"] += 1
            if sac is not None:
                b["sac_num"] += sac * wt
                b["sac_denom"] += wt
            if res is not None:
                b["res_num"] += res * wt
                b["res_denom"] += wt

    out = {}
    for alpha, b in agg.items():
        if b["sac_denom"] > 0 and b["res_denom"] > 0:
            out[alpha] = {
                "sacsecval": b["sac_num"] / b["sac_denom"],
                "resemaval":  b["res_num"] / b["res_denom"],
                "n": b["n_rows"],
            }
    return out


def aggregate_all_waves() -> dict[str, dict]:
    """Merge all available waves, most-recent-wins per country."""
    merged: dict[str, dict] = {}
    for fname, delim, alpha_col, num_col in WVS_WAVES:
        path = WVS_DIR / fname
        if not path.exists():
            print(f"  skipping {fname} (not found)", file=sys.stderr)
            continue
        print(f"  reading {fname}...", file=sys.stderr)
        wave_data = aggregate_wave(path, delim, alpha_col, num_col)
        new_count = 0
        for iso, d in wave_data.items():
            if iso not in merged:
                merged[iso] = d
                new_count += 1
        print(f"    {len(wave_data)} countries, {new_count} new", file=sys.stderr)
    return merged


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
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    print("Aggregating WVS waves (most-recent-wins per country)...", file=sys.stderr)
    countries = aggregate_all_waves()
    print(f"Total: {len(countries)} countries with coordinates", file=sys.stderr)

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

    # Inject NormAd-only countries that have no WVS data — use cluster centroid.
    injected = []
    for iso, cluster in NORMAD_ONLY.items():
        if iso in countries:
            continue  # WVS data already present, skip
        if cluster not in centroids:
            print(f"  WARNING: no centroid for {cluster} (needed by {iso}), skipping",
                  file=sys.stderr)
            continue
        cx, cy = centroids[cluster]
        countries[iso] = {
            "sacsecval": cx,
            "resemaval": cy,
            "n": 0,
            "cluster": cluster,
            "cluster_inferred": False,
        }
        injected.append(iso)
    if injected:
        print(f"\nInjected {len(injected)} NormAd-only countries with cluster centroid: "
              f"{', '.join(sorted(injected))}", file=sys.stderr)

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

    # Write output CSV, sorted by distance ascending (US-similar first).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(countries.items(), key=lambda kv: kv[1]["dist_from_english"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country_iso", "normad_country", "cluster", "cluster_inferred",
                    "sacsecval", "resemaval", "dist_from_english", "n_respondents"])
        for alpha, d in rows:
            w.writerow([
                alpha,
                NORMAD_NAME.get(alpha, ""),
                d["cluster"],
                "yes" if d.get("cluster_inferred") else "no",
                f"{d['sacsecval']:.4f}",
                f"{d['resemaval']:.4f}",
                f"{d['dist_from_english']:.4f}",
                d["n"],
            ])
    print(f"\nWrote {out_path} ({len(rows)} countries)", file=sys.stderr)

    # Print per-cluster summary to stderr.
    print(f"\nCluster summary (mean dist from EnglishSpeaking centroid):", file=sys.stderr)
    by_cluster = defaultdict(list)
    for alpha, d in countries.items():
        by_cluster[d["cluster"]].append(d["dist_from_english"])
    for cluster in sorted(by_cluster, key=lambda c: sum(by_cluster[c]) / len(by_cluster[c])):
        ds = by_cluster[cluster]
        print(f"  {cluster:18s}  mean dist = {sum(ds)/len(ds):.3f}  (n={len(ds)} countries)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
