"""Print per-cluster country lists and centroid distances from English-Speaking,
for manual validation of the I-W classification produced by compute_iw_coords.py.

Input:  data/iw_coordinates.csv  (produced by analysis/culturemapping/compute_iw_coords.py)
Outputs:
  - stdout: human-readable summary by cluster
  - data/iw_clusters_summary.csv: machine-readable summary (one row per cluster)

Usage:
    python3 analysis/culturemapping/validate_iw_clusters.py
    python3 analysis/culturemapping/validate_iw_clusters.py --sort-by alphabetical
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORDS_CSV = PROJECT_ROOT / "data" / "iw_coordinates.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "iw_clusters_summary.csv"


def load_coords(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["sacsecval"] = float(r["sacsecval"])
            r["resemaval"] = float(r["resemaval"])
            r["dist_from_english"] = float(r["dist_from_english"])
            r["n_respondents"] = int(r["n_respondents"])
            r["cluster_inferred"] = r.get("cluster_inferred", "no") == "yes"
            rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", default=str(COORDS_CSV))
    parser.add_argument("--summary-out", default=str(SUMMARY_CSV))
    parser.add_argument(
        "--sort-by", choices=["distance", "alphabetical"], default="distance",
        help="Sort countries within each cluster by distance to anglosphere "
             "(default) or alphabetically by ISO code.",
    )
    args = parser.parse_args()

    coords_path = Path(args.coords)
    if not coords_path.exists():
        print(f"ERROR: {coords_path} not found. Run analysis/culturemapping/compute_iw_coords.py first.",
              file=sys.stderr)
        sys.exit(1)

    rows = load_coords(coords_path)
    if not rows:
        print(f"ERROR: {coords_path} is empty", file=sys.stderr)
        sys.exit(1)

    # Group by cluster
    by_cluster = defaultdict(list)
    for r in rows:
        by_cluster[r["cluster"]].append(r)

    # Cluster centroid = weighted mean of (sacsecval, resemaval) using n_respondents
    centroids = {}
    for cluster, members in by_cluster.items():
        total_n = sum(m["n_respondents"] for m in members)
        cx = sum(m["sacsecval"] * m["n_respondents"] for m in members) / total_n
        cy = sum(m["resemaval"] * m["n_respondents"] for m in members) / total_n
        centroids[cluster] = (cx, cy, total_n)

    # English-Speaking centroid is the anchor for inter-cluster distances
    eng = centroids.get("EnglishSpeaking")
    if eng is None:
        print("ERROR: no EnglishSpeaking cluster in coords file", file=sys.stderr)
        sys.exit(2)
    eng_cx, eng_cy, _ = eng

    cluster_dist = {}
    for cluster, (cx, cy, _) in centroids.items():
        cluster_dist[cluster] = math.hypot(cx - eng_cx, cy - eng_cy)

    # === STDOUT: per-cluster summary ===
    sorted_clusters = sorted(cluster_dist.items(), key=lambda kv: kv[1])

    print(f"\nI-W cluster validation — anchor: English-Speaking centroid "
          f"({eng_cx:.3f}, {eng_cy:.3f})\n")

    for cluster, d in sorted_clusters:
        cx, cy, total_n = centroids[cluster]
        members = by_cluster[cluster]
        inferred_count = sum(1 for m in members if m["cluster_inferred"])

        print(f"\n========== {cluster} ==========")
        print(f"  centroid: ({cx:.3f}, {cy:.3f})")
        print(f"  distance from EnglishSpeaking centroid: {d:.3f}")
        print(f"  n countries: {len(members)} ({inferred_count} inferred via nearest-centroid)")
        print(f"  n respondents (sum): {total_n:,}")
        print(f"\n  Countries (sorted by {'distance' if args.sort_by == 'distance' else 'alpha-3'}):")

        sort_key = (lambda x: x["dist_from_english"]) if args.sort_by == "distance" \
                   else (lambda x: x["country_iso"])
        for m in sorted(members, key=sort_key):
            marker = " *INFERRED*" if m["cluster_inferred"] else ""
            normad = f"  (normad: {m['normad_country']})" if m['normad_country'] else ""
            print(f"    {m['country_iso']:>4s}  dist={m['dist_from_english']:.3f}  "
                  f"sac={m['sacsecval']:.3f}  res={m['resemaval']:.3f}"
                  f"{normad}{marker}")

    # === MACHINE-READABLE SUMMARY CSV ===
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cluster", "centroid_sacsecval", "centroid_resemaval",
            "dist_from_english_centroid", "n_countries", "n_respondents",
            "n_inferred", "countries_iso", "countries_normad",
        ])
        for cluster, d in sorted_clusters:
            cx, cy, total_n = centroids[cluster]
            members = by_cluster[cluster]
            sorted_members = sorted(members, key=lambda m: m["country_iso"])
            iso_list = ";".join(m["country_iso"] for m in sorted_members)
            normad_list = ";".join(m["normad_country"] for m in sorted_members if m["normad_country"])
            n_inferred = sum(1 for m in members if m["cluster_inferred"])
            w.writerow([
                cluster, f"{cx:.4f}", f"{cy:.4f}", f"{d:.4f}",
                len(members), total_n, n_inferred,
                iso_list, normad_list,
            ])
    print(f"\nWrote summary CSV: {summary_path}\n")

    # === TOP-LEVEL TABLE ===
    print(f"\n{'='*70}")
    print(f"  Cluster centroids — distance from EnglishSpeaking, ordered ascending")
    print(f"{'='*70}")
    print(f"  {'cluster':<18s}  {'dist_from_eng':>13s}  {'n_countries':>11s}  {'n_resp':>10s}")
    for cluster, d in sorted_clusters:
        n_countries = len(by_cluster[cluster])
        _, _, total_n = centroids[cluster]
        print(f"  {cluster:<18s}  {d:>13.3f}  {n_countries:>11d}  {total_n:>10,}")


if __name__ == "__main__":
    main()
