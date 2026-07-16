"""Bar graph: NormAd accuracy by US-similarity binary, per condition.

For each of the 9 conditions, two bars:
  - LEFT bar  = mean accuracy across countries in {EnglishSpeaking, ProtestantEurope}
                ("US-similar" — anglosphere + nearest European cluster)
  - RIGHT bar = mean accuracy across countries in every other I-W cluster
                ("US-distant" — Catholic Europe + Latin America + Orthodox + Confucian + South Asia + African-Islamic)

Single figure with X = condition, grouped pair of bars per condition. Shows how
alignment training changes the US-similar vs US-distant accuracy gap.

Reads:
  data/iw_coordinates.csv               ← per-country I-W cluster
  outputs/behavioral/normad_*.json      ← per-condition predictions

Outputs:
  outputs/figures/accuracy_bars_us_similar_vs_distant.pdf

Usage:
    python3 analysis/cluster_accuracy_bars.py
    python3 analysis/cluster_accuracy_bars.py --exclude dpo
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SETUP_CONDITIONS = {
    "all": ["base", "sft", "dpo", "sftdpo"],
}

COND_LABELS = {
    "base":   "C1\nBase",
    "sft":    "C2\nSFT",
    "dpo":    "C3\nDPO",
    "sftdpo": "C4\nSFT+DPO",
}

# Definition of the binary US-similarity split. Edit to taste — e.g., add
# CatholicEurope to the US-similar side, or restrict US-similar to just
# EnglishSpeaking.
US_SIMILAR_CLUSTERS = {"EnglishSpeaking", "ProtestantEurope"}

COLOR_US_SIMILAR = "#4477AA"   # blue
COLOR_US_DISTANT = "#CC6677"   # warm red


def load_country_to_cluster(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run analysis/culturemapping/compute_iw_coords.py first.")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["normad_country"]:
                out[r["normad_country"]] = r["cluster"]
    return out


def accuracy_by_group(cond: str, country_to_cluster: dict, size_suffix: str = "") -> dict[str, tuple[float, int, float]]:
    """Return {group: (accuracy, n_predictions, sem)} for one condition.

    SEM is the standard error of accuracy across COUNTRIES (not predictions) in
    the group, giving a sense of within-group variability.
    """
    path = BEHAVIORAL_DIR / f"normad_{cond}{size_suffix}.json"
    if not path.exists():
        return {}
    preds = json.loads(path.read_text()).get("predictions", [])

    # First: per-country (correct, total)
    per_country = defaultdict(lambda: [0, 0])
    for p in preds:
        country = p.get("country", "")
        cluster = country_to_cluster.get(country) or country_to_cluster.get(country.lower())
        if cluster is None:
            continue
        per_country[(country, cluster)][0] += int(p["gold"] == p["pred"])
        per_country[(country, cluster)][1] += 1

    # Then: group-level aggregation
    group_totals = defaultdict(lambda: [0, 0])    # group → [correct, total]
    group_country_accs = defaultdict(list)         # group → list of per-country accuracies
    for (country, cluster), (c, t) in per_country.items():
        group = "US-similar" if cluster in US_SIMILAR_CLUSTERS else "US-distant"
        group_totals[group][0] += c
        group_totals[group][1] += t
        group_country_accs[group].append(c / t)

    result = {}
    for group, (c, t) in group_totals.items():
        accs = group_country_accs[group]
        # SEM across countries (not predictions) — captures cluster-internal variability
        sem = (np.std(accs, ddof=1) / math.sqrt(len(accs))) if len(accs) > 1 else 0.0
        result[group] = (c / t, t, sem)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=list(SETUP_CONDITIONS), default="all")
    parser.add_argument("--exclude", nargs="+", default=[])
    parser.add_argument("--model-size", choices=["3b", "8b", "gemma4", "qwen35"], default="3b")
    parser.add_argument("--no-errorbars", action="store_true",
                        help="Hide SEM error bars on top of each bar.")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in SETUP_CONDITIONS[args.setup] if c not in args.exclude]
    if not conditions:
        sys.exit("No conditions left after --exclude")

    size_suffix = "" if args.model_size == "3b" else f"_{args.model_size}"
    fig_size_suffix = f"_{args.model_size}"
    suffix = "" if (args.setup == "all" and not args.exclude) else f"_{args.setup}"
    if args.exclude:
        suffix += "_no_" + "_".join(sorted(args.exclude))
    suffix += fig_size_suffix

    country_to_cluster = load_country_to_cluster(IW_COORDS)

    # Compute per-condition per-group accuracies
    sim_accs, sim_sems, sim_ns = [], [], []
    dist_accs, dist_sems, dist_ns = [], [], []
    for cond in conditions:
        groups = accuracy_by_group(cond, country_to_cluster, size_suffix)
        s = groups.get("US-similar", (np.nan, 0, 0.0))
        d = groups.get("US-distant", (np.nan, 0, 0.0))
        sim_accs.append(s[0]); sim_ns.append(s[1]); sim_sems.append(s[2])
        dist_accs.append(d[0]); dist_ns.append(d[1]); dist_sems.append(d[2])

    # Plot
    x = np.arange(len(conditions))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(conditions)), 5))

    # ax.bar's error-bar styling goes inside error_kw, not as top-level kwargs.
    err_kw = dict(error_kw=dict(capsize=3, ecolor="gray", elinewidth=1)) \
        if not args.no_errorbars else {}
    bars_sim = ax.bar(x - width / 2, sim_accs, width,
                      yerr=sim_sems if not args.no_errorbars else None,
                      color=COLOR_US_SIMILAR, edgecolor="black", linewidth=0.5,
                      label="US-similar (EnglishSpeaking + ProtestantEurope)",
                      **err_kw)
    bars_dist = ax.bar(x + width / 2, dist_accs, width,
                       yerr=dist_sems if not args.no_errorbars else None,
                       color=COLOR_US_DISTANT, edgecolor="black", linewidth=0.5,
                       label="US-distant (all other clusters)",
                       **err_kw)

    # Numeric value labels on top of each bar
    for bar, acc, n in zip(bars_sim, sim_accs, sim_ns):
        if not math.isnan(acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.3f}\n(n={n})", ha="center", va="bottom", fontsize=7)
    for bar, acc, n in zip(bars_dist, dist_accs, dist_ns):
        if not math.isnan(acc):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{acc:.3f}\n(n={n})", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=9)
    ax.set_ylabel("NormAd accuracy")
    ax.set_ylim(0, max(max(sim_accs + [0]), max(dist_accs + [0])) * 1.18)
    ax.set_title("NormAd accuracy: US-similar vs US-distant cluster groups, per condition")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = FIGURES_DIR / f"accuracy_bars_us_similar_vs_distant{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Console summary too — useful for paste-into-paper
    print(f"\n  {'condition':<18s}  {'US-similar':>12s}  {'US-distant':>12s}  {'gap':>10s}")
    for cond, sa, da in zip(conditions, sim_accs, dist_accs):
        gap = sa - da if not (math.isnan(sa) or math.isnan(da)) else float("nan")
        print(f"  {cond:<18s}  {sa:>12.3f}  {da:>12.3f}  {gap:>+10.3f}")


if __name__ == "__main__":
    main()
