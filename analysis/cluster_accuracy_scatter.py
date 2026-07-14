"""Per-condition scatter plots: NormAd accuracy vs I-W cluster distance from US.

For each model condition, plots 8 points — one per Inglehart-Welzel cluster —
with X = distance of the cluster centroid from the English-Speaking centroid
and Y = mean NormAd accuracy across countries in that cluster. Overlays
Spearman correlation (with p-value) and an OLS regression line so the
cultural gradient is readable at a glance.

Reads:
  data/iw_coordinates.csv               ← per-country I-W cluster + coords
  outputs/behavioral/normad_*.json      ← per-condition predictions

Outputs:
  outputs/figures/cluster_accuracy_scatter.pdf       (multi-panel grid, all conds)
  outputs/figures/cluster_accuracy_scatter_<cond>.pdf (one per condition, with --per-condition)

Usage:
    python3 analysis/cluster_accuracy_scatter.py                     # all 4 conds, one grid
    python3 analysis/cluster_accuracy_scatter.py --exclude dpo
    python3 analysis/cluster_accuracy_scatter.py --per-condition     # one PDF per condition
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
    "base":   "C1: Base",
    "sft":    "C2: SFT",
    "dpo":    "C3: DPO",
    "sftdpo": "C4: SFT+DPO",
}

# Consistent cluster colors across all panels.
CLUSTER_COLORS = {
    "EnglishSpeaking": "#4477AA",
    "ProtestantEurope": "#66CCEE",
    "CatholicEurope":  "#228833",
    "LatinAmerica":    "#CCBB44",
    "Orthodox":        "#EE7733",
    "Confucian":       "#CC3311",
    "SouthAsia":       "#AA3377",
    "AfricanIslamic":  "#000000",
}


def load_iw_coords(path: Path):
    """Return (country_to_cluster dict, cluster_to_dist dict)."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run analysis/culturemapping/compute_iw_coords.py first.")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["normad_country"] or not r["sacsecval"]:
                continue
            rows.append({
                "country": r["normad_country"],
                "cluster": r["cluster"],
                "sacsecval": float(r["sacsecval"]),
                "resemaval": float(r["resemaval"]),
                "n_respondents": int(r["n_respondents"]),
            })

    country_to_cluster = {r["country"]: r["cluster"] for r in rows}

    # Cluster centroid = weighted mean of (sacsecval, resemaval) using n_respondents.
    centroids = defaultdict(lambda: {"sx": 0.0, "sy": 0.0, "w": 0})
    for r in rows:
        n = r["n_respondents"]
        centroids[r["cluster"]]["sx"] += r["sacsecval"] * n
        centroids[r["cluster"]]["sy"] += r["resemaval"] * n
        centroids[r["cluster"]]["w"] += n
    cluster_centroid = {
        c: (b["sx"] / b["w"], b["sy"] / b["w"])
        for c, b in centroids.items() if b["w"] > 0
    }

    eng = cluster_centroid.get("EnglishSpeaking")
    if eng is None:
        sys.exit("ERROR: no EnglishSpeaking cluster in coords file")
    cluster_to_dist = {
        c: math.hypot(x - eng[0], y - eng[1])
        for c, (x, y) in cluster_centroid.items()
    }
    return country_to_cluster, cluster_to_dist


def per_cluster_accuracy(cond: str, country_to_cluster: dict, size_suffix: str = "") -> dict[str, tuple[float, int]]:
    """Return {cluster: (accuracy, n)} for one condition's NormAd JSON."""
    path = BEHAVIORAL_DIR / f"normad_{cond}{size_suffix}.json"
    if not path.exists():
        return {}
    preds = json.loads(path.read_text()).get("predictions", [])
    by_cluster = defaultdict(lambda: [0, 0])  # [correct, total]
    for p in preds:
        # NormAd country names may differ slightly in case; try both.
        country = p.get("country", "")
        cluster = country_to_cluster.get(country) or country_to_cluster.get(country.lower())
        if cluster is None:
            continue
        by_cluster[cluster][0] += int(p["gold"] == p["pred"])
        by_cluster[cluster][1] += 1
    return {c: (correct / total, total) for c, (correct, total) in by_cluster.items() if total > 0}


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float | None]:
    """Spearman rho + p-value. Uses scipy if available; falls back to rho-only."""
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(xs, ys)
        return float(rho), float(p)
    except ImportError:
        # Manual Spearman without p-value
        if len(xs) < 2:
            return float("nan"), None
        rx = _ranks(xs)
        ry = _ranks(ys)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return (num / den if den > 0 else float("nan")), None


def _ranks(xs):
    """Average ranks of xs (handles ties)."""
    pairs = sorted(enumerate(xs), key=lambda kv: kv[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg
        i = j + 1
    return ranks


def draw_panel(ax, cond: str, cluster_to_dist: dict[str, float],
               cluster_acc: dict[str, tuple[float, int]]):
    """Draw one scatter panel: clusters as colored dots with regression line + stats."""
    points = []  # (cluster, x, y, n)
    for cluster, dist in cluster_to_dist.items():
        if cluster in cluster_acc:
            acc, n = cluster_acc[cluster]
            points.append((cluster, dist, acc, n))
    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(COND_LABELS.get(cond, cond), fontsize=10)
        return

    for cluster, x, y, n in points:
        ax.scatter(x, y, s=20 + 2 * math.sqrt(n), color=CLUSTER_COLORS.get(cluster, "gray"),
                   edgecolor="black", linewidth=0.5, alpha=0.85, zorder=3)

    # OLS regression line
    xs = np.array([p[1] for p in points])
    ys = np.array([p[2] for p in points])
    if len(xs) >= 2:
        m, b = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(x_line, m * x_line + b, "--", color="gray", alpha=0.6, linewidth=1)

    # Spearman annotation
    rho, p = spearman(list(xs), list(ys))
    if p is not None:
        sig = "*" if p < 0.05 else ""
        ax.text(0.02, 0.97, f"ρ={rho:+.2f}  p={p:.2f}{sig}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))
    else:
        ax.text(0.02, 0.97, f"ρ={rho:+.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))

    ax.set_title(COND_LABELS.get(cond, cond), fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(left=-0.05)


def draw_combined(conditions: list[str], country_to_cluster, cluster_to_dist,
                  out_path: Path, size_suffix: str = ""):
    """Multi-panel grid: one scatter per condition, shared axes."""
    n = len(conditions)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for i, cond in enumerate(conditions):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        cluster_acc = per_cluster_accuracy(cond, country_to_cluster, size_suffix)
        draw_panel(ax, cond, cluster_to_dist, cluster_acc)
        if c == 0:
            ax.set_ylabel("NormAd accuracy")
        if r == rows - 1:
            ax.set_xlabel("Distance from EnglishSpeaking centroid")
    # Hide unused panels
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    # Shared legend for cluster colors
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=CLUSTER_COLORS[c], markeredgecolor="black",
                          markersize=8, label=c)
               for c in CLUSTER_COLORS if c in cluster_to_dist]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("NormAd accuracy by I-W cluster across conditions", y=0.99, fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def draw_per_condition(conditions: list[str], country_to_cluster, cluster_to_dist,
                       suffix: str, size_suffix: str = ""):
    """One PDF per condition."""
    for cond in conditions:
        cluster_acc = per_cluster_accuracy(cond, country_to_cluster, size_suffix)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        draw_panel(ax, cond, cluster_to_dist, cluster_acc)
        ax.set_xlabel("Distance from EnglishSpeaking centroid")
        ax.set_ylabel("NormAd accuracy")
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=CLUSTER_COLORS[c], markeredgecolor="black",
                              markersize=8, label=c)
                   for c in CLUSTER_COLORS if c in cluster_to_dist]
        ax.legend(handles=handles, loc="best", fontsize=7, frameon=True)
        fig.tight_layout()
        out = FIGURES_DIR / f"cluster_accuracy_scatter_{cond}{suffix}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=list(SETUP_CONDITIONS), default="all")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="Conditions to drop from the setup, e.g. --exclude dpo")
    parser.add_argument("--per-condition", action="store_true",
                        help="Emit one PDF per condition instead of a single grid")
    parser.add_argument("--model-size", choices=["3b", "8b", "gemma4"], default="3b")
    args = parser.parse_args()

    size_suffix = "" if args.model_size == "3b" else f"_{args.model_size}"
    fig_size_suffix = f"_{args.model_size}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in SETUP_CONDITIONS[args.setup] if c not in args.exclude]
    if not conditions:
        sys.exit("No conditions left after --exclude")

    suffix = "" if (args.setup == "all" and not args.exclude) else f"_{args.setup}"
    if args.exclude:
        suffix += "_no_" + "_".join(sorted(args.exclude))
    suffix += fig_size_suffix

    country_to_cluster, cluster_to_dist = load_iw_coords(IW_COORDS)

    if args.per_condition:
        draw_per_condition(conditions, country_to_cluster, cluster_to_dist, suffix, size_suffix)
    else:
        out = FIGURES_DIR / f"cluster_accuracy_scatter{suffix}.pdf"
        draw_combined(conditions, country_to_cluster, cluster_to_dist, out, size_suffix)


if __name__ == "__main__":
    main()
