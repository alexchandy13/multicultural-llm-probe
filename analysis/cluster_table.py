"""Export cluster-level neuron stats to CSV.

Two files written to outputs/figures/cluster_csvs/:

  cluster_mean_activation.csv
    Rows = conditions, columns = I-W clusters (ascending dist from EnglishSpeaking).
    Cell = mean activation score of all culture neurons for countries in that cluster.
    (Same data as the cluster_activation heatmap, in table form.)

  cluster_top_neuron_count.csv
    Rows = conditions, columns = I-W clusters.
    Cell = number of culture neurons whose highest mean per-country activation is
    in that cluster (exclusive assignment — each neuron counted once).

Usage:
    python analysis/cluster_table.py                        # 8b normad yn
    python analysis/cluster_table.py --model-size 3b        # 3b normad (no yn suffix)
    python analysis/cluster_table.py --conditions base sft_aya_cult
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_DIR   = PROJECT_ROOT / "outputs" / "neurons"
FIGURES_DIR   = PROJECT_ROOT / "outputs" / "figures"
IW_COORDS     = PROJECT_ROOT / "data" / "iw_coordinates.csv"

ALL_CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult", "sftdpo_aya_nocult"]


def load_iw_coords():
    rows = []
    with open(IW_COORDS) as f:
        for r in csv.DictReader(f):
            if not r["normad_country"] or not r["dist_from_english"]:
                continue
            rows.append({
                "country": r["normad_country"],
                "cluster": r["cluster"],
                "dist":    float(r["dist_from_english"]),
            })
    country_to_cluster = {r["country"]: r["cluster"] for r in rows}
    cluster_dists = defaultdict(list)
    for r in rows:
        cluster_dists[r["cluster"]].append(r["dist"])
    ordered = sorted(cluster_dists, key=lambda c: np.mean(cluster_dists[c]))
    return country_to_cluster, ordered


def load_neurons(cond_dir: Path, yn_only: bool) -> list[dict]:
    suffix = "_yn" if yn_only else ""
    p = cond_dir / f"all_neurons_normad{suffix}_max.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("top_neurons", [])


def load_scores(cond_dir: Path, yn_only: bool) -> dict:
    name = "normad_yn_max_scores.json" if yn_only else "normad_max_scores.json"
    p = cond_dir / name
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("neuron_scores", {})


def compute(cond_dir: Path, yn_only: bool, country_to_cluster: dict, cluster_order: list):
    neurons = load_neurons(cond_dir, yn_only)
    scores  = load_scores(cond_dir, yn_only)
    if not neurons or not scores:
        return None, None

    # per-cluster mean activation
    per_cluster: dict[str, list[float]] = defaultdict(list)
    # per-neuron cluster means (for top-cluster assignment)
    neuron_cluster_means: list[dict[str, float]] = []

    for n in neurons:
        key = f"{n['module_name']}_{n['layer_idx']}_{n['neuron_idx']}"
        country_scores = scores.get(key, {})
        if not country_scores:
            neuron_cluster_means.append({})
            continue

        nc: dict[str, list[float]] = defaultdict(list)
        for raw_country, score in country_scores.items():
            cluster = (country_to_cluster.get(raw_country)
                       or country_to_cluster.get(raw_country.lower())
                       or country_to_cluster.get(raw_country.title()))
            if cluster:
                nc[cluster].append(score)
                per_cluster[cluster].append(score)

        neuron_cluster_means.append({c: float(np.mean(v)) for c, v in nc.items()})

    mean_activation = {
        c: float(np.mean(per_cluster[c])) if c in per_cluster else float("nan")
        for c in cluster_order
    }

    top_cluster_counts: dict[str, int] = {c: 0 for c in cluster_order}
    for ncm in neuron_cluster_means:
        if not ncm:
            continue
        top = max(ncm, key=ncm.__getitem__)
        top_cluster_counts[top] = top_cluster_counts.get(top, 0) + 1

    return mean_activation, top_cluster_counts


def write_csv(path: Path, rows: list[tuple[str, dict]], cluster_order: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition"] + cluster_order)
        for cond, data in rows:
            w.writerow([cond] + [
                f"{data[c]:.6f}" if isinstance(data[c], float) and not np.isnan(data[c])
                else ("" if isinstance(data[c], float) else str(data[c]))
                for c in cluster_order
            ])
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["3b", "8b", "gemma4", "qwen35"], default="8b")
    parser.add_argument("--yn-only", action="store_true", default=True,
                        help="Use normad_yn files (default: True for 8b).")
    parser.add_argument("--no-yn-only", dest="yn_only", action="store_false")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    args = parser.parse_args()

    size_suffix = "" if args.model_size == "3b" else f"_{args.model_size}"

    country_to_cluster, cluster_order = load_iw_coords()

    mean_rows, count_rows = [], []
    for cond in args.conditions:
        cond_dir = NEURONS_DIR / f"{cond}{size_suffix}"
        if not cond_dir.exists():
            print(f"[skip] {cond_dir} not found")
            continue
        mean_act, top_counts = compute(cond_dir, args.yn_only, country_to_cluster, cluster_order)
        if mean_act is None:
            print(f"[skip] no data for {cond}")
            continue
        mean_rows.append((cond, mean_act))
        count_rows.append((cond, top_counts))

    out_dir = FIGURES_DIR / "cluster_csvs"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = size_suffix or "_3b"

    write_csv(out_dir / f"cluster_mean_activation{tag}.csv", mean_rows, cluster_order)
    write_csv(out_dir / f"cluster_top_neuron_count{tag}.csv", count_rows, cluster_order)


if __name__ == "__main__":
    main()
