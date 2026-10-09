"""Per-layer count of cluster-specific neurons: one figure per condition, 8 lines.

Reconstructs the Stage B (z-filtered, cluster-specific) sets from
cluster_membership_*.json rather than reading the eight per-cluster selection
files. The membership file stores every candidate neuron with its layer, its
cluster memberships and its per-cluster z-scores, so "specific to C" is
recoverable as (C in clusters) and (zscores[C] >= threshold) — identical to what
Stage B writes, and it means one 34 MB file per condition instead of eight.

It also makes the threshold a plot-time parameter: --zscore-threshold re-slices
without re-running any selection.

One figure per condition rather than one per cluster, because the question is
whether a given checkpoint concentrates cluster-specific neurons at a particular
depth, and whether the eight clusters agree about where that is. Lines are
coloured per cluster and the colour is fixed across figures, so the ten figures
can be flipped through and compared directly.

    python analysis/cluster_specific_layer_lines.py --model-size 8b
    python analysis/cluster_specific_layer_lines.py --model-size 8b --normalize
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from culnig.clusters import CLUSTERS

NEURONS_ROOT = PROJECT_ROOT / "outputs" / "neurons"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "cluster_specific_layer_counts"

CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult",
              "sftdpo_aya_cult", "sftdpo_aya_nocult"]

# Fixed per cluster so colour means the same thing in every figure. Warm hues for
# the Western-adjacent clusters, cool for the rest, which makes the Western/
# non-Western split readable without reading the legend.
CLUSTER_COLORS = {
    "EnglishSpeaking":  "#b2182b",
    "ProtestantEurope": "#ef8a62",
    "CatholicEurope":   "#f4a582",
    "Orthodox":         "#998ec3",
    "Confucian":        "#2166ac",
    "SouthAsia":        "#4393c3",
    "AfricanIslamic":   "#1b7837",
    "LatinAmerica":     "#762a83",
}


def load_counts(cond_dir: Path, suffix: str, zthr: float):
    """{cluster: {layer: count}} plus the highest layer index seen.

    A neuron counts toward cluster C when it is a C candidate (Stage A) and its
    C z-score clears the threshold (Stage B).
    """
    path = cond_dir / f"cluster_membership_{suffix}_max.json"
    if not path.exists():
        return {}, -1
    d = json.load(open(path))
    eligible = d["clusters"]
    counts: dict[str, dict[int, int]] = {cl: defaultdict(int) for cl in eligible}
    max_layer = -1
    for n in d["neurons"]:
        lay = int(n["layer_idx"])
        max_layer = max(max_layer, lay)
        zs = n.get("zscores", {})
        for cl in n["clusters"]:
            if cl in counts and zs.get(cl, 0.0) >= zthr:
                counts[cl][lay] += 1
    return {cl: dict(v) for cl, v in counts.items() if v}, max_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="8b",
                    choices=["3b", "8b", "gemma4", "qwen35"])
    ap.add_argument("--dataset", default="normad")
    ap.add_argument("--yn-only", action="store_true", default=True)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--normalize", action="store_true",
                    help="Plot each cluster's share of its own total instead of raw "
                         "counts. Clusters select different numbers of neurons, so "
                         "raw counts conflate 'where' with 'how many'.")
    ap.add_argument("--log-y", action="store_true")
    ap.add_argument("--zscore-threshold", type=float, default=0.5,
                    help="Stage B threshold, 0.5 to match decide_cluster_neurons.py. "
                         "Pass a large negative number to plot Stage A candidates "
                         "instead, i.e. culture neurons per cluster with no "
                         "specificity filter.")
    args = ap.parse_args()

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    suffix = args.dataset + ("_yn" if args.yn_only else "")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    wrote = []
    for cond in args.conditions:
        cond_dir = NEURONS_ROOT / f"{cond}{size_sfx}"
        counts, max_layer = load_counts(cond_dir, suffix, args.zscore_threshold)
        if not counts:
            print(f"  {cond}{size_sfx}: no cluster_membership_{suffix}_max.json — skipped")
            continue
        n_layers = max_layer + 1

        fig, ax = plt.subplots(figsize=(10, 5))
        for cl in CLUSTERS:
            if cl not in counts:
                continue
            ys = [counts[cl].get(l, 0) for l in range(n_layers)]
            tot = sum(ys) or 1
            if args.normalize:
                ys = [100 * y / tot for y in ys]
            ax.plot(range(n_layers), ys, color=CLUSTER_COLORS[cl],
                    linewidth=1.8, label=f"{cl} (n={tot})")

        ax.set_xlabel("Layer")
        ax.set_ylabel("% of cluster's neurons" if args.normalize
                      else "# cluster-specific neurons")
        ax.set_title(f"Cluster-specific neurons per layer — "
                     f"{cond}{size_sfx}, {suffix}")
        if args.log_y:
            ax.set_yscale("log")
        ax.set_xlim(0, n_layers - 1)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(fontsize=8, title="IW cluster", title_fontsize=8, ncol=2)
        fig.tight_layout()

        ztag = "" if args.zscore_threshold == 0.5 else f"_z{args.zscore_threshold:g}"
        tag = ztag + ("_norm" if args.normalize else "") + ("_log" if args.log_y else "")
        out = FIGURES_DIR / f"line_cluster_specific_count_{cond}{size_sfx}_{suffix}{tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        wrote.append(out)
        print(f"  {cond}{size_sfx}: {n_layers} layers, "
              f"{len(counts)} clusters, {sum(sum(v.values()) for v in counts.values())} neurons "
              f"-> {out.name}")

    print(f"\n{len(wrote)} figure(s) in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
