"""Line graph of NormAd accuracy across an alignment pipeline, by I-W cluster.

Default mode ("clusters"): one line for the merged US-similar group
(EnglishSpeaking + ProtestantEurope), plus one line per other I-W cluster
(CatholicEurope, Orthodox, Confucian, LatinAmerica, SouthAsia, AfricanIslamic,
…). Cluster order in the legend = ascending mean distance from the English-
Speaking centroid, so the legend reads roughly "closest → furthest" from
anglosphere baseline.

Alternative mode ("binary"): the original two-line plot — US-similar vs
US-distant — preserved behind --mode binary.

X axis = conditions in order (default: Base → SFT-Alp → SFT+DPO-Alp), so the
slope between adjacent points = the per-stage Δaccuracy for that cluster.

Default pipeline matches the paper's alpaca framing:
  Base → SFT-Alp → SFT+DPO-Alp

Override via --conditions. Examples:
  python3 analysis/accuracy_pipeline_lines.py --conditions base sft dpo sftdpo
  python3 analysis/accuracy_pipeline_lines.py --mode binary
  python3 analysis/accuracy_pipeline_lines.py --conditions base sft_lima sftdpo_lima

Reads:
  data/iw_coordinates.csv               ← per-country I-W cluster (+ distance)
  outputs/behavioral/normad_*.json      ← per-condition predictions

Outputs:
  outputs/figures/accuracy_pipeline_lines{suffix}.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

US_SIMILAR_CLUSTERS = {"EnglishSpeaking", "ProtestantEurope"}
US_SIMILAR_LABEL = "US-similar"   # merged-group key used in cluster mode

COND_LABELS = {
    "base":           "Base",
    "sft":            "SFT-HH",
    "sft_alpaca":     "SFT",
    "sft_lima":       "SFT-LIMA",
    "dpo":            "DPO",
    "sftdpo":         "SFT-HH+DPO",
    "sftdpo_alpaca":  "SFT+DPO",
    "sftdpo_lima":    "SFT+DPO-LIMA",
    "instruct":       "Instruct",
}

DEFAULT_PIPELINE = ["base", "sft_alpaca", "sftdpo_alpaca"]

COLOR_US_SIMILAR = "#4477AA"   # blue — reserved for the merged US-similar group
COLOR_US_DISTANT = "#CC6677"   # warm red — only used in binary mode

# Tol "muted" palette minus blue (reserved for US-similar). Used round-robin
# over the non-US-similar clusters; with 6 typical clusters we won't wrap.
PALETTE_OTHER = [
    "#CC6677",  # red
    "#117733",  # green
    "#DDCC77",  # yellow
    "#882255",  # wine
    "#88CCEE",  # cyan
    "#999933",  # olive
    "#AA4499",  # purple
    "#332288",  # indigo
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]


def load_iw_data(path: Path) -> tuple[dict[str, str], dict[str, float]]:
    """Return (country->cluster, cluster->mean_dist_from_english)."""
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run analysis/compute_iw_coords.py first.")
    country_to_cluster: dict[str, str] = {}
    cluster_dists: dict[str, list[float]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["normad_country"]:
                continue
            country_to_cluster[r["normad_country"]] = r["cluster"]
            try:
                cluster_dists[r["cluster"]].append(float(r["dist_from_english"]))
            except (KeyError, ValueError):
                pass
    cluster_mean_dist = {c: float(np.mean(d)) for c, d in cluster_dists.items()}
    return country_to_cluster, cluster_mean_dist


def pooled_accuracy_by_group(
    cond: str,
    country_to_cluster: dict[str, str],
    mode: str,
) -> dict[str, float]:
    """Return {group_label: accuracy} for one condition.

    `mode='binary'` -> groups are {'US-similar','US-distant'}.
    `mode='clusters'` -> groups are {'US-similar', '<cluster_name>', ...}
                         where every non-US-similar cluster gets its own key.

    Accuracy is pooled across all predictions in each group (matching the
    convention in accuracy_deltas_bars.py and the original two-line plot).
    """
    path = BEHAVIORAL_DIR / f"normad_{cond}.json"
    if not path.exists():
        return {}
    preds = json.loads(path.read_text()).get("predictions", [])
    by_group: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in preds:
        country = p.get("country", "")
        cluster = (country_to_cluster.get(country)
                   or country_to_cluster.get(country.lower()))
        if cluster is None:
            continue
        if mode == "binary":
            group = US_SIMILAR_LABEL if cluster in US_SIMILAR_CLUSTERS else "US-distant"
        else:  # "clusters"
            group = US_SIMILAR_LABEL if cluster in US_SIMILAR_CLUSTERS else cluster
        by_group[group][0] += int(p["gold"] == p["pred"])
        by_group[group][1] += 1
    return {g: (c / t) for g, (c, t) in by_group.items() if t > 0}


def order_groups(group_keys: set[str], cluster_mean_dist: dict[str, float],
                 mode: str) -> list[str]:
    """US-similar first, then remaining clusters by ascending mean distance.

    In binary mode this is just ['US-similar', 'US-distant'].
    """
    if mode == "binary":
        order = []
        if US_SIMILAR_LABEL in group_keys:
            order.append(US_SIMILAR_LABEL)
        if "US-distant" in group_keys:
            order.append("US-distant")
        return order
    # clusters mode
    rest = [g for g in group_keys if g != US_SIMILAR_LABEL]
    rest.sort(key=lambda c: cluster_mean_dist.get(c, float("inf")))
    return ([US_SIMILAR_LABEL] if US_SIMILAR_LABEL in group_keys else []) + rest


def color_for(group: str, idx_in_order: int) -> str:
    """US-similar always blue; every other group draws from PALETTE_OTHER in
    the order it appears in the legend, so colors are deterministic."""
    if group == US_SIMILAR_LABEL:
        return COLOR_US_SIMILAR
    # Map idx_in_order==0 -> first non-US-similar slot; account for the fact
    # that US-similar may or may not be present in the order.
    return PALETTE_OTHER[idx_in_order % len(PALETTE_OTHER)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["clusters", "binary"], default="clusters",
        help="'clusters' (default): one line per I-W cluster, with EnglishSpeaking "
             "and ProtestantEurope merged into a single 'US-similar' line. "
             "'binary': original two-line US-similar vs US-distant plot.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Ordered list of conditions to plot on the X axis. "
             "Default: base sft_alpaca sftdpo_alpaca.",
    )
    parser.add_argument("--no-values", action="store_true",
                        help="Hide numeric accuracy annotations next to each point. "
                             "Defaults: ON in binary mode, OFF (no labels) in clusters "
                             "mode since 7 lines × 3 points is too cluttered to label.")
    parser.add_argument("--values", action="store_true",
                        help="Force-show value annotations even in clusters mode.")
    parser.add_argument("--y-from-zero", action="store_true",
                        help="Force Y axis to start at 0 (default: auto-zoomed to data range).")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to the output PDF filename.")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions if args.conditions else DEFAULT_PIPELINE
    if len(conditions) < 2:
        sys.exit("Need at least 2 conditions for a line graph.")

    country_to_cluster, cluster_mean_dist = load_iw_data(IW_COORDS)

    # Compute per-condition per-group accuracies
    # acc_by_group: {group_label -> [acc_at_cond_0, acc_at_cond_1, ...]}
    acc_by_group: dict[str, list[float]] = defaultdict(
        lambda: [float("nan")] * len(conditions)
    )
    for j, cond in enumerate(conditions):
        groups = pooled_accuracy_by_group(cond, country_to_cluster, args.mode)
        if not groups:
            print(f"[warn] no data for condition {cond!r}; will appear as NaN",
                  file=sys.stderr)
        for g, acc in groups.items():
            acc_by_group[g][j] = acc

    # Decide legend order + assign colors
    ordered_groups = order_groups(set(acc_by_group.keys()), cluster_mean_dist,
                                  args.mode)
    color_map: dict[str, str] = {}
    other_idx = 0
    for g in ordered_groups:
        if g == US_SIMILAR_LABEL:
            color_map[g] = COLOR_US_SIMILAR
        elif args.mode == "binary":
            color_map[g] = COLOR_US_DISTANT
        else:
            color_map[g] = PALETTE_OTHER[other_idx % len(PALETTE_OTHER)]
            other_idx += 1

    # Annotation policy: ON in binary mode by default (matches the original
    # script's behaviour); OFF in clusters mode because 7 lines × ≥3 points
    # plus value labels turns into noise. --values forces ON; --no-values
    # forces OFF.
    if args.no_values:
        show_values = False
    elif args.values:
        show_values = True
    else:
        show_values = (args.mode == "binary")

    # === PLOT ===
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(conditions)), 5.5))

    for i, group in enumerate(ordered_groups):
        accs = acc_by_group[group]
        # US-similar gets a slightly thicker line so it's distinguishable as
        # the "merged anglosphere baseline" reference line in the cluster plot.
        lw = 2.6 if group == US_SIMILAR_LABEL else 2.0
        ms = 9 if group == US_SIMILAR_LABEL else 7
        # Build legend label
        if group == US_SIMILAR_LABEL:
            label = "US-Centric/Western"
        elif args.mode == "binary":
            label = "US-Distant/Non-Western"
        else:
            d = cluster_mean_dist.get(group)
            label = f"{group}" + (f"  (d={d:.2f})" if d is not None else "")
        ax.plot(x, accs,
                marker=MARKERS[i % len(MARKERS)],
                markersize=ms, linewidth=lw,
                color=color_map[group],
                label=label)

        if show_values:
            for xi, v in zip(x, accs):
                if not math.isnan(v):
                    offset = (0, 10) if group == US_SIMILAR_LABEL else (0, -15)
                    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                                xytext=offset, ha="center", fontsize=9,
                                color=color_map[group])

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=11)
    ax.set_xlabel("Alignment Condition")
    ax.set_ylabel("NormAd Accuracy", fontsize=11)
    if args.mode == "binary":
        title = "NormAd Accuracy Across Alignment Pipeline: Western vs Non-Western"
    else:
        title = ("NormAd accuracy across alignment pipeline, by I-W cluster\n"
                 "(EnglishSpeaking + ProtestantEurope merged; legend ordered by "
                 "distance from anglosphere centroid)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    # In cluster mode the legend is wide — pull it outside the axes so it
    # doesn't sit on top of the lines.
    if args.mode == "binary":
        ax.legend(loc="best", fontsize=10, frameon=True)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=True)

    # Y range
    finite = [v for accs in acc_by_group.values() for v in accs if not math.isnan(v)]
    if finite:
        if args.y_from_zero:
            ax.set_ylim(0, max(finite) * 1.10)
        else:
            lo, hi = min(finite), max(finite)
            span = max(hi - lo, 0.02)
            ax.set_ylim(lo - span * 0.15, hi + span * 0.15)

    ax.set_xlim(-0.4, len(conditions) - 0.6)

    fig.tight_layout()
    out = FIGURES_DIR / f"accuracy_pipeline_lines{args.out_suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # === Console summary — handy for paste-into-paper ===
    header = f"  {'condition':<18s}  " + "  ".join(
        f"{g[:16]:>16s}" for g in ordered_groups
    )
    print(f"\n{header}")
    for j, cond in enumerate(conditions):
        row = f"  {cond:<18s}  " + "  ".join(
            f"{acc_by_group[g][j]:>16.3f}" if not math.isnan(acc_by_group[g][j])
            else f"{'nan':>16s}"
            for g in ordered_groups
        )
        print(row)


if __name__ == "__main__":
    main()
