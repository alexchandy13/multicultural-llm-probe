"""CULNIG-style heatmaps for visualizing culture-neuron distribution across conditions.

Four heatmaps, all written to outputs/figures/:

  1. heatmap_layer_count.pdf
     Y: 28 layers × X: conditions. Cell = number of identified culture neurons
     at that layer in that condition. Answers: where in the model do culture
     neurons live, and does alignment shift the distribution?

  2. heatmap_layer_attribution.pdf
     Y: 28 layers × X: conditions. Cell = mean attribute_score of culture
     neurons at that layer. Answers: how strongly are culture-relevant layers
     activated across alignment stages? This is the "fading" visualization
     from the original CULNIG paper.

  3. heatmap_module_distribution.pdf
     Y: target modules (mlp.gate_proj, self_attn.q_proj, etc.) × X: conditions.
     Cell = count or mean attribution. Answers: which architectural components
     host the culture representation?

  4. heatmap_cluster_activation.pdf
     Y: 8 Inglehart-Welzel clusters (ordered by distance from EnglishSpeaking)
     × X: conditions. Cell = mean culture-neuron activation across countries in
     that cluster. Answers: do culture neurons activate more strongly for
     US-similar cultures, and does alignment widen that gradient?

Usage:
    python3 analysis/heatmaps.py                       # all 4 figures, --setup all
    python3 analysis/heatmaps.py --figures cluster     # just the cluster heatmap
    python3 analysis/heatmaps.py --setup hhrlhf        # only the primary 5 conditions
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_DIR = PROJECT_ROOT / "outputs" / "neurons"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"

SETUP_CONDITIONS = {
    "hhrlhf": ["base", "sft", "dpo", "sftdpo", "instruct"],
    "alpaca": ["base", "sft_alpaca", "dpo", "sftdpo_alpaca", "instruct"],
    "lima":   ["base", "sft_lima", "dpo", "sftdpo_lima", "instruct"],
    "both":   ["base", "sft", "sft_alpaca", "dpo", "sftdpo", "sftdpo_alpaca", "instruct"],
    "all":    ["base", "sft", "sft_alpaca", "sft_lima", "dpo",
               "sftdpo", "sftdpo_alpaca", "sftdpo_lima", "instruct"],
}

COND_LABELS = {
    "base":           "C1\nBase",
    "sft":            "C2\nSFT",
    "sft_alpaca":     "C2a\nSFT-Alp",
    "sft_lima":       "C2b\nSFT-LIMA",
    "dpo":            "C3\nDPO",
    "sftdpo":         "C4\nSFT+DPO",
    "sftdpo_alpaca":  "C4a\nSFT(Alp)+DPO",
    "sftdpo_lima":    "C4b\nSFT(LIMA)+DPO",
    "instruct":       "C5\nInstruct",
}

# Order modules sensibly (MLP first, then attention) for the per-module heatmap.
MODULE_ORDER = [
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
]


def load_neurons(cond: str) -> list[dict] | None:
    """Return the list of selected culture neurons for a condition, or None if missing."""
    path = NEURONS_DIR / cond / "all_neurons_normad_max.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("top_neurons", [])


def load_per_country_scores(cond: str) -> dict | None:
    """Return raw per-neuron per-country scores from normad_max_scores.json."""
    path = NEURONS_DIR / cond / "normad_max_scores.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("neuron_scores", {})


def load_iw_coords(path: Path) -> tuple[dict[str, str], list[str]] | None:
    """Return (normad_country -> cluster) dict + cluster order (ascending dist from English)."""
    if not path.exists():
        print(f"[warn] {path} not found — cluster heatmap will be skipped", file=sys.stderr)
        return None
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r["normad_country"]:
                continue
            rows.append({
                "country": r["normad_country"],
                "cluster": r["cluster"],
                "dist": float(r["dist_from_english"]),
            })
    country_to_cluster = {r["country"]: r["cluster"] for r in rows}
    # Order clusters by mean dist from EnglishSpeaking centroid
    cluster_dists = defaultdict(list)
    for r in rows:
        cluster_dists[r["cluster"]].append(r["dist"])
    ordered = sorted(cluster_dists, key=lambda c: np.mean(cluster_dists[c]))
    return country_to_cluster, ordered


def figure_layer_count(conditions: list[str], suffix: str):
    """Heatmap of culture-neuron count per (layer × condition)."""
    n_layers = 28  # Llama 3.2 3B
    matrix = np.full((n_layers, len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        neurons = load_neurons(cond)
        if neurons is None:
            continue
        counts = defaultdict(int)
        for n in neurons:
            counts[n["layer_idx"]] += 1
        for layer, c in counts.items():
            matrix[layer, j] = c

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(conditions)), 7))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Layer")
    ax.set_title("Culture-neuron count per layer × condition")
    plt.colorbar(im, ax=ax, label="# culture neurons")
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_layer_count{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_layer_attribution(conditions: list[str], suffix: str):
    """Heatmap of mean attribute_score per (layer × condition) — the 'fading' picture."""
    n_layers = 28
    matrix = np.full((n_layers, len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        neurons = load_neurons(cond)
        if neurons is None:
            continue
        per_layer = defaultdict(list)
        for n in neurons:
            per_layer[n["layer_idx"]].append(n["attribute_score"])
        for layer, scores in per_layer.items():
            matrix[layer, j] = float(np.mean(scores))

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(conditions)), 7))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Layer")
    ax.set_title("Mean culture-neuron attribution score per layer × condition")
    plt.colorbar(im, ax=ax, label="Mean (NormAd − NormAdctrl) attribution score")
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_layer_attribution{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_module_distribution(conditions: list[str], suffix: str):
    """Heatmap of culture-neuron count per (module × condition)."""
    matrix = np.full((len(MODULE_ORDER), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        neurons = load_neurons(cond)
        if neurons is None:
            continue
        counts = defaultdict(int)
        for n in neurons:
            counts[n["module_name"]] += 1
        for i, module in enumerate(MODULE_ORDER):
            if module in counts:
                matrix[i, j] = counts[module]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(conditions)), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper")
    ax.set_yticks(range(len(MODULE_ORDER)))
    ax.set_yticklabels(MODULE_ORDER, fontsize=9)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Module")
    ax.set_title("Culture-neuron count per module × condition")
    plt.colorbar(im, ax=ax, label="# culture neurons")
    # Annotate cell values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, int(v), ha="center", va="center",
                        color="white" if v < np.nanmax(matrix) / 2 else "black",
                        fontsize=7)
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_module_distribution{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_cluster_activation(conditions: list[str], suffix: str):
    """Heatmap of mean culture-neuron activation per (I-W cluster × condition).

    For each condition: take the identified culture neurons from
    all_neurons_normad_max.json, look up their per-country activations from
    normad_max_scores.json, group countries by I-W cluster (loaded from
    data/iw_coordinates.csv), and average within each cluster.
    """
    iw = load_iw_coords(IW_COORDS)
    if iw is None:
        return
    country_to_cluster, cluster_order = iw

    matrix = np.full((len(cluster_order), len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        neurons = load_neurons(cond)
        scores = load_per_country_scores(cond)
        if neurons is None or scores is None:
            continue
        per_cluster = defaultdict(list)
        for n in neurons:
            key = f"{n['module_name']}_{n['layer_idx']}_{n['neuron_idx']}"
            if key not in scores:
                continue
            for country, score in scores[key].items():
                # Map "Country" → normad lowercase name if possible (handle case mismatches)
                norm_country = country.lower() if isinstance(country, str) else country
                # Try both raw and lowercased lookup
                cluster = country_to_cluster.get(country) or country_to_cluster.get(norm_country)
                if cluster:
                    per_cluster[cluster].append(score)
        for i, cluster in enumerate(cluster_order):
            if cluster in per_cluster:
                matrix[i, j] = float(np.mean(per_cluster[cluster]))

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(conditions)), max(4, 0.6 * len(cluster_order))))
    im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper")
    ax.set_yticks(range(len(cluster_order)))
    ax.set_yticklabels(cluster_order, fontsize=9)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax.set_xlabel("Condition")
    ax.set_ylabel("I-W cluster (ordered by distance from EnglishSpeaking, ascending)")
    ax.set_title("Mean culture-neuron activation per I-W cluster × condition")
    plt.colorbar(im, ax=ax, label="Mean activation score")
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_cluster_activation{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setup", choices=list(SETUP_CONDITIONS), default="all",
        help="Which condition set to include. Default: all (9 conditions).",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=[],
        help="Condition names to drop from the chosen --setup. "
             "E.g., --exclude sftdpo_lima to skip C4b.",
    )
    parser.add_argument(
        "--figures", nargs="+",
        choices=["layer_count", "layer_attribution", "module_distribution",
                 "cluster_activation", "all"],
        default=["all"],
        help="Which heatmaps to generate. Default: all four.",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in SETUP_CONDITIONS[args.setup] if c not in args.exclude]
    # Suffix encodes both setup and exclusions so figures don't overwrite.
    suffix = "" if (args.setup == "all" and not args.exclude) else f"_{args.setup}"
    if args.exclude:
        suffix += "_no_" + "_".join(sorted(args.exclude))

    todo = ({"layer_count", "layer_attribution", "module_distribution", "cluster_activation"}
            if "all" in args.figures else set(args.figures))

    if "layer_count" in todo:        figure_layer_count(conditions, suffix)
    if "layer_attribution" in todo:  figure_layer_attribution(conditions, suffix)
    if "module_distribution" in todo: figure_module_distribution(conditions, suffix)
    if "cluster_activation" in todo:  figure_cluster_activation(conditions, suffix)


if __name__ == "__main__":
    main()
