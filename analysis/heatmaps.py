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

  5. heatmap_group_attribution.pdf  (--figures group_attribution)
     Two-panel figure: layer × condition. Top panel = mean per-neuron
     attribution restricted to US-similar countries (EnglishSpeaking +
     ProtestantEurope); bottom panel = US-distant. Both panels share vmin/vmax
     so the visual brightness is directly comparable. Answers: under which
     conditions does attribution flow symmetrically vs. asymmetrically across
     the western/non-western split? Direct neuron-level counterpart of the
     US-similar/US-distant accuracy bars.

  6a. heatmap_group_attribution_by_module.pdf
      (--figures group_attribution_by_module)
      Same recipe as (5), but rows are the 7 target modules (mlp.gate_proj,
      mlp.up_proj, mlp.down_proj, self_attn.q_proj/k_proj/v_proj/o_proj)
      instead of layers. Answers the architectural-component version of the
      same question: does the western/non-western attribution split live in
      MLP neurons or attention neurons?

  6. heatmap_asymmetry_attribution.pdf  (--figures asymmetry_attribution)
     Single-panel: layer × condition. Cell = (US-similar mean) − (US-distant
     mean) of per-neuron attribution. Diverging colormap centered at 0. Red =
     attribution biased toward western countries at that (layer, condition);
     blue = biased toward non-western. Designed to make SFT+DPO's western/
     non-western divergence pop visually even when the absolute attribution
     magnitudes shift across conditions.

Usage:
    python3 analysis/heatmaps.py                       # all 4 figures, --setup all
    python3 analysis/heatmaps.py --figures cluster_activation
    python3 analysis/heatmaps.py --setup all           # all 4 conditions

    # Layer-count heatmap, sequential pipeline only
    python3 analysis/heatmaps.py --figures layer_count \\
        --conditions base sft sftdpo --out-suffix _pipeline

    # Layer-count heatmap, SFT-alone vs DPO-alone
    python3 analysis/heatmaps.py --figures layer_count \\
        --conditions sft dpo --out-suffix _sft_vs_dpo

    # Group-split attribution heatmaps for the SFT-vs-DPO-vs-SFT+DPO story
    python3 analysis/heatmaps.py \\
        --figures group_attribution asymmetry_attribution \\
        --conditions base sft dpo sftdpo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEURONS_DIR = PROJECT_ROOT / "outputs" / "neurons"
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"

# Same definition used by the cluster_accuracy_bars / accuracy_deltas_bars
# scripts, so the heatmap split matches the behavioural bar plots exactly.
US_SIMILAR_CLUSTERS = {"EnglishSpeaking", "ProtestantEurope"}

SETUP_CONDITIONS = {
    "all": ["base", "sft", "dpo", "sftdpo"],
}

COND_LABELS = {
    "base":   "Base",
    "sft":    "SFT",
    "dpo":    "DPO",
    "sftdpo": "SFT+DPO",
}

# The modules that this project's CULNIG pipeline actually saves as culture
# neurons. Restricted by:
#   - culnig/_upstream/CULNIG/calc_neuron_score.py:TARGET_MODULES
#     (scores 5 modules — adds mlp.up_proj — but never down_proj or o_proj)
#   - culnig/decide_culture_neurons_normad.py:MLP_SAVE_MODULES / ATTENTION_SAVE_MODULES
#     (further drops mlp.up_proj from the saved selection)
# So the modules that appear in all_neurons_normad_max.json are exactly these 4.
# Ordered MLP-first, then attention Q/K/V in the natural QKV order.
MODULE_ORDER = [
    "mlp.gate_proj",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
]


def _setup_layer_axis(ax, matrix: np.ndarray, conditions: list[str],
                      n_layers: int, transpose: bool) -> np.ndarray:
    """Apply layer/condition tick labels in the right orientation.

    Returns the matrix that should be passed to imshow — transposed if needed.
    The input `matrix` is always shaped (n_layers, len(conditions)); the
    transposed version is (len(conditions), n_layers).
    """
    if transpose:
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Condition")
        return matrix.T
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Layer")
    return matrix


def _layer_axis_figsize(n_conditions: int, transpose: bool,
                        n_layers: int = 28, base_height: float = 7.0,
                        panels: int = 1) -> tuple[float, float]:
    """Return a (width, height) figsize tailored to the transpose state.

    Normal layout: condition on X, layer on Y -> tall, narrow.
    Transposed:    layer on X, condition on Y -> wide, short.
    For multi-panel figures (group_attribution), `panels` scales the height
    in the normal layout and the height in the transposed layout symmetrically.
    """
    if transpose:
        # 28 layers along X — give each ~0.3" of horizontal space.
        width = max(8, 0.32 * n_layers)
        # Conditions along Y — give each ~0.7" plus padding for ticks/colorbars.
        height = max(2.5 * panels, 0.7 * n_conditions * panels + 1.5)
        return (width, height)
    return (max(7, 0.9 * n_conditions), base_height * panels)


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


def load_per_country_control_scores(cond: str) -> dict | None:
    """Return raw per-neuron per-country scores from normadcontrol_max_scores.json.

    Same shape as normad_max_scores, computed on the control prompts. Used to
    perform a per-country version of the (normad − normadcontrol) subtraction
    that decide_culture_neurons_normad.py does at the all-countries-collapsed
    level when computing the saved scalar `attribute_score`.
    """
    path = NEURONS_DIR / cond / "normadcontrol_max_scores.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("neuron_scores", {})


def per_country_item_counts() -> dict[str, int]:
    """Count NormAd items per country, using the base condition's behavioral
    output as ground truth (NormAd country labels are condition-independent).

    Used to normalize the per-country sums in normad_max_scores.json into a
    per-item mean — otherwise countries with more NormAd items would dominate
    the heatmap purely on item-count grounds.
    """
    # Prefer base; fall back to any available condition. We just need the
    # country→item-count map, which is invariant across conditions.
    candidates = ["base"] + [d.name for d in sorted(BEHAVIORAL_DIR.glob("normad_*.json"))]
    counts: dict[str, int] = defaultdict(int)
    for stem in candidates:
        path = BEHAVIORAL_DIR / f"normad_{stem}.json" if not stem.endswith(".json") else BEHAVIORAL_DIR / stem
        if not path.exists():
            continue
        for p in json.loads(path.read_text()).get("predictions", []):
            country = p.get("country", "")
            if country:
                counts[country] += 1
        if counts:
            return dict(counts)
    return dict(counts)


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


def figure_layer_count(conditions: list[str], suffix: str, transpose: bool = False):
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

    fig, ax = plt.subplots(figsize=_layer_axis_figsize(len(conditions), transpose))
    display = _setup_layer_axis(ax, matrix, conditions, n_layers, transpose)
    im = ax.imshow(display, aspect="auto", cmap="magma", origin="upper")
    ax.set_title("Culture-neuron count per layer × condition")
    plt.colorbar(im, ax=ax, label="# culture neurons")
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_layer_count{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_layer_attribution(conditions: list[str], suffix: str, transpose: bool = False):
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

    fig, ax = plt.subplots(figsize=_layer_axis_figsize(len(conditions), transpose))
    display = _setup_layer_axis(ax, matrix, conditions, n_layers, transpose)
    im = ax.imshow(display, aspect="auto", cmap="magma", origin="upper")
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


def _compute_layer_group_attribution(
    cond: str,
    country_to_cluster: dict[str, str],
    item_counts: dict[str, int],
    subtract_control: bool,
    n_layers: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """For one condition, return two length-n_layers arrays:
       (mean per-neuron attribution averaged over US-similar countries,
        same for US-distant), with NaN where no data.

    Recipe:
      1. Take the condition's *selected* culture neurons.
      2. For each neuron, look up its per-country attribution sum from
         normad_max_scores.json. Optionally subtract control_scores[key].
         Divide by the per-country NormAd item count to get a per-item mean.
      3. Average those per-item means across countries within each group, then
         across all culture neurons that live at the same layer.
    """
    sim = np.full(n_layers, np.nan)
    dist = np.full(n_layers, np.nan)

    neurons = load_neurons(cond)
    scores = load_per_country_scores(cond)
    ctrl = load_per_country_control_scores(cond) if subtract_control else None
    if neurons is None or scores is None:
        print(f"[warn] missing neuron data for {cond!r}; column will be NaN",
              file=sys.stderr)
        return sim, dist

    # per-layer accumulators of per-neuron group-means
    per_layer_sim: dict[int, list[float]] = defaultdict(list)
    per_layer_dist: dict[int, list[float]] = defaultdict(list)

    for n in neurons:
        key = f"{n['module_name']}_{n['layer_idx']}_{n['neuron_idx']}"
        country_scores = scores.get(key)
        if not country_scores:
            continue
        ctrl_scores = ctrl.get(key, {}) if ctrl is not None else {}

        # accumulate per-country per-item attributions split by group
        sim_vals, dist_vals = [], []
        for raw_country, raw_score in country_scores.items():
            # Country labels in score dicts use mixed casing; try direct then lower.
            cluster = (country_to_cluster.get(raw_country)
                       or country_to_cluster.get(raw_country.lower())
                       or country_to_cluster.get(raw_country.title()))
            if cluster is None:
                continue
            n_items = (item_counts.get(raw_country)
                       or item_counts.get(raw_country.lower())
                       or item_counts.get(raw_country.title()))
            if not n_items:
                continue
            ctrl_score = ctrl_scores.get(raw_country, 0.0) if ctrl is not None else 0.0
            per_item_attr = (raw_score - ctrl_score) / n_items
            if cluster in US_SIMILAR_CLUSTERS:
                sim_vals.append(per_item_attr)
            else:
                dist_vals.append(per_item_attr)

        layer = n["layer_idx"]
        if sim_vals:
            per_layer_sim[layer].append(float(np.mean(sim_vals)))
        if dist_vals:
            per_layer_dist[layer].append(float(np.mean(dist_vals)))

    for layer, vals in per_layer_sim.items():
        sim[layer] = float(np.mean(vals))
    for layer, vals in per_layer_dist.items():
        dist[layer] = float(np.mean(vals))
    return sim, dist


def figure_group_attribution(conditions: list[str], suffix: str,
                             subtract_control: bool, transpose: bool = False):
    """Two-panel heatmap: layer × condition, US-similar (top) and US-distant (bottom).

    Each cell = mean per-item attribution of this condition's culture neurons
    at this layer, averaged over countries in the group. Both panels share
    vmin/vmax so visual brightness is directly comparable between groups.
    The story you want to read off the figure: under SFT alone and DPO alone
    both panels brighten symmetrically; under SFT+DPO the top stays bright
    while the bottom dims (or vice versa), matching the behavioural finding.
    """
    iw = load_iw_coords(IW_COORDS)
    if iw is None:
        return
    country_to_cluster, _ = iw
    item_counts = per_country_item_counts()
    if not item_counts:
        print("[warn] per-country item counts unavailable; "
              "skipping group_attribution heatmap", file=sys.stderr)
        return

    n_layers = 28
    sim_matrix = np.full((n_layers, len(conditions)), np.nan)
    dist_matrix = np.full((n_layers, len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        sim, dist = _compute_layer_group_attribution(
            cond, country_to_cluster, item_counts, subtract_control, n_layers
        )
        sim_matrix[:, j] = sim
        dist_matrix[:, j] = dist

    # Shared color range across both panels (ignoring NaN)
    stacked = np.concatenate([sim_matrix.ravel(), dist_matrix.ravel()])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        print("[warn] no finite values for group_attribution; skipping",
              file=sys.stderr)
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    figsize = _layer_axis_figsize(len(conditions), transpose, panels=2,
                                   base_height=5.5)
    if transpose:
        # Layers along X → stack the two panels vertically (shared X = layers)
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        # Conditions along X → keep stacked vertically with shared X (matches
        # the original layout)
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for ax, matrix, title in [
        (axes[0], sim_matrix, "US-Centric/Western"),
        (axes[1], dist_matrix, "US-Distant/Non-Western"),
    ]:
        display = _setup_layer_axis(ax, matrix, conditions, n_layers, transpose)
        im = ax.imshow(display, aspect="auto", cmap="magma", origin="upper",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="Mean per-item Attribution")
    sub = " (normad − normadcontrol)" if subtract_control else " (normad raw)"
    fig.suptitle("Per-neuron Attribution by Group" + sub, fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_group_attribution{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_group_attribution_per_condition(conditions: list[str], suffix: str,
                                            subtract_control: bool):
    """One mini-panel per condition; each panel is a 2-row layer-heatmap with
    US-Centric/Western on top and US-Distant/Non-Western on the bottom.

    Layout: N stacked panels (one per condition), each panel is a (2, n_layers)
    heatmap. All panels share vmin/vmax so brightness is directly comparable
    across conditions. Compared to figure_group_attribution() this view makes
    it easier to read "for THIS condition, how does attribution flow across
    layers, sim vs dist" rather than "for THIS layer, how does sim or dist
    flow across conditions".
    """
    iw = load_iw_coords(IW_COORDS)
    if iw is None:
        return
    country_to_cluster, _ = iw
    item_counts = per_country_item_counts()
    if not item_counts:
        print("[warn] per-country item counts unavailable; "
              "skipping group_attribution_per_condition", file=sys.stderr)
        return

    n_layers = 28
    per_cond: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond in conditions:
        sim, dist = _compute_layer_group_attribution(
            cond, country_to_cluster, item_counts, subtract_control, n_layers
        )
        per_cond[cond] = (sim, dist)

    # Shared color range across all conditions
    all_vals = []
    for sim, dist in per_cond.values():
        all_vals.extend(sim[np.isfinite(sim)].tolist())
        all_vals.extend(dist[np.isfinite(dist)].tolist())
    if not all_vals:
        print("[warn] no finite values for group_attribution_per_condition; "
              "skipping", file=sys.stderr)
        return
    vmin = float(min(all_vals))
    vmax = float(max(all_vals))

    n_panels = len(conditions)
    # Each panel: layer along X (~0.32" per cell), 2 rows of cells (sim, dist).
    # Per-panel height ~1.4" gives room for the two cells + title bar.
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(max(10, 0.34 * n_layers), 1.6 * n_panels + 1.0),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    im = None
    for ax, cond in zip(axes, conditions):
        sim, dist = per_cond[cond]
        matrix = np.vstack([sim, dist])  # shape (2, n_layers)
        im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper",
                       vmin=vmin, vmax=vmax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["US-Centric/\nWestern", "US-Distant/\nNon-Western"],
                          fontsize=8)
        # Condition label as left-aligned title so it sits next to the panel
        ax.set_title(COND_LABELS.get(cond, cond), fontsize=11, loc="left")

    # Only the bottom panel shows the full layer-axis tick labels.
    axes[-1].set_xticks(range(n_layers))
    axes[-1].set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=7)
    axes[-1].set_xlabel("Layer")
    # Hide x-tick labels on the upper panels (sharex shows them by default).
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    # One shared colorbar on the right covering all panels.
    fig.colorbar(im, ax=axes.tolist(), location="right",
                 label="Mean per-item attribution", shrink=0.85)

    sub = " (normad − normadcontrol)" if subtract_control else " (normad raw)"
    fig.suptitle(f"Per-condition layer attribution by group{sub}", fontsize=12)
    out = FIGURES_DIR / f"heatmap_group_attribution_per_condition{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_group_attribution_per_condition_by_module(
        conditions: list[str], suffix: str, subtract_control: bool):
    """Module-axis counterpart to figure_group_attribution_per_condition.

    Layout: N stacked panels (one per condition), each panel is a
    (2, n_modules) heatmap with US-Centric/Western on top and US-Distant/
    Non-Western on the bottom. Shared colorbar across all panels. With only
    4 modules in MODULE_ORDER, each panel is short and wide-cell rather than
    the long depth-profile of the per-layer version.

    Reads from the same {module → per-group mean per-item attribution} dicts
    produced by _compute_module_group_attribution(), so this is consistent
    with figure_group_attribution_by_module().
    """
    iw = load_iw_coords(IW_COORDS)
    if iw is None:
        return
    country_to_cluster, _ = iw
    item_counts = per_country_item_counts()
    if not item_counts:
        print("[warn] per-country item counts unavailable; "
              "skipping group_attribution_per_condition_by_module",
              file=sys.stderr)
        return

    n_mods = len(MODULE_ORDER)
    per_cond: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond in conditions:
        sim_dict, dist_dict = _compute_module_group_attribution(
            cond, country_to_cluster, item_counts, subtract_control
        )
        sim = np.array([sim_dict.get(m, float("nan")) for m in MODULE_ORDER])
        dist = np.array([dist_dict.get(m, float("nan")) for m in MODULE_ORDER])
        per_cond[cond] = (sim, dist)

    # Shared color range across all conditions
    all_vals = []
    for sim, dist in per_cond.values():
        all_vals.extend(sim[np.isfinite(sim)].tolist())
        all_vals.extend(dist[np.isfinite(dist)].tolist())
    if not all_vals:
        print("[warn] no finite values for group_attribution_per_condition_by_module; "
              "skipping", file=sys.stderr)
        return
    vmin = float(min(all_vals))
    vmax = float(max(all_vals))

    n_panels = len(conditions)
    # With only 4 modules, give each cell ~1.4" of horizontal space so the
    # module names fit underneath. Each panel ~1.6" tall (2 cells + title).
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(max(7, 1.5 * n_mods + 2.0), 1.7 * n_panels + 1.0),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    im = None
    for ax, cond in zip(axes, conditions):
        sim, dist = per_cond[cond]
        matrix = np.vstack([sim, dist])  # shape (2, n_mods)
        im = ax.imshow(matrix, aspect="auto", cmap="magma", origin="upper",
                       vmin=vmin, vmax=vmax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["US-Centric/\nWestern", "US-Distant/\nNon-Western"],
                          fontsize=8)
        ax.set_title(COND_LABELS.get(cond, cond), fontsize=11, loc="left")

    # Only the bottom panel shows the module-axis tick labels.
    axes[-1].set_xticks(range(n_mods))
    axes[-1].set_xticklabels(MODULE_ORDER, fontsize=9, rotation=20, ha="right")
    axes[-1].set_xlabel("Module")
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.colorbar(im, ax=axes.tolist(), location="right",
                 label="Mean per-item attribution", shrink=0.85)

    sub = " (normad − normadcontrol)" if subtract_control else " (normad raw)"
    fig.suptitle(f"Per-condition module attribution by group{sub}", fontsize=12)
    out = FIGURES_DIR / f"heatmap_group_attribution_per_condition_by_module{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _compute_module_group_attribution(
    cond: str,
    country_to_cluster: dict[str, str],
    item_counts: dict[str, int],
    subtract_control: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """For one condition, return two {module → mean per-neuron attribution}
    dicts: one averaged over US-similar countries, one over US-distant.

    Identical to _compute_layer_group_attribution() except the per-neuron
    group-means are bucketed by module_name instead of layer_idx. A neuron's
    contribution is averaged across all layers that host it for the same
    module — which is what you want when asking "do MLP gate_proj neurons
    behave differently from attention v_proj neurons w.r.t. western/non-western
    attribution?", independent of where in depth they live.
    """
    sim: dict[str, float] = {m: float("nan") for m in MODULE_ORDER}
    dist: dict[str, float] = {m: float("nan") for m in MODULE_ORDER}

    neurons = load_neurons(cond)
    scores = load_per_country_scores(cond)
    ctrl = load_per_country_control_scores(cond) if subtract_control else None
    if neurons is None or scores is None:
        print(f"[warn] missing neuron data for {cond!r}; column will be NaN",
              file=sys.stderr)
        return sim, dist

    per_mod_sim: dict[str, list[float]] = defaultdict(list)
    per_mod_dist: dict[str, list[float]] = defaultdict(list)

    for n in neurons:
        key = f"{n['module_name']}_{n['layer_idx']}_{n['neuron_idx']}"
        country_scores = scores.get(key)
        if not country_scores:
            continue
        ctrl_scores = ctrl.get(key, {}) if ctrl is not None else {}

        sim_vals, dist_vals = [], []
        for raw_country, raw_score in country_scores.items():
            cluster = (country_to_cluster.get(raw_country)
                       or country_to_cluster.get(raw_country.lower())
                       or country_to_cluster.get(raw_country.title()))
            if cluster is None:
                continue
            n_items = (item_counts.get(raw_country)
                       or item_counts.get(raw_country.lower())
                       or item_counts.get(raw_country.title()))
            if not n_items:
                continue
            ctrl_score = ctrl_scores.get(raw_country, 0.0) if ctrl is not None else 0.0
            per_item_attr = (raw_score - ctrl_score) / n_items
            if cluster in US_SIMILAR_CLUSTERS:
                sim_vals.append(per_item_attr)
            else:
                dist_vals.append(per_item_attr)

        mod = n["module_name"]
        if sim_vals:
            per_mod_sim[mod].append(float(np.mean(sim_vals)))
        if dist_vals:
            per_mod_dist[mod].append(float(np.mean(dist_vals)))

    for m, vals in per_mod_sim.items():
        sim[m] = float(np.mean(vals))
    for m, vals in per_mod_dist.items():
        dist[m] = float(np.mean(vals))
    return sim, dist


def figure_group_attribution_by_module(conditions: list[str], suffix: str,
                                        subtract_control: bool,
                                        transpose: bool = False):
    """Two-panel heatmap: module × condition, US-similar (top) and US-distant (bottom).

    Direct architectural-component analog of figure_group_attribution. Rows are
    the 7 target modules in MODULE_ORDER (MLP first, then attention). Both
    panels share vmin/vmax so brightness is comparable across groups.

    If --transpose is on, module runs along X and condition along Y — useful
    for paper figures that want a wide-banner aspect ratio.
    """
    iw = load_iw_coords(IW_COORDS)
    if iw is None:
        return
    country_to_cluster, _ = iw
    item_counts = per_country_item_counts()
    if not item_counts:
        print("[warn] per-country item counts unavailable; "
              "skipping group_attribution_by_module heatmap", file=sys.stderr)
        return

    n_mods = len(MODULE_ORDER)
    sim_matrix = np.full((n_mods, len(conditions)), np.nan)
    dist_matrix = np.full((n_mods, len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        sim, dist = _compute_module_group_attribution(
            cond, country_to_cluster, item_counts, subtract_control
        )
        for i, m in enumerate(MODULE_ORDER):
            sim_matrix[i, j] = sim.get(m, np.nan)
            dist_matrix[i, j] = dist.get(m, np.nan)

    stacked = np.concatenate([sim_matrix.ravel(), dist_matrix.ravel()])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        print("[warn] no finite values for group_attribution_by_module; skipping",
              file=sys.stderr)
        return
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    # Module-axis figure has only 7 rows, so it doesn't need the tall layout
    # of the 28-layer version. Sized for two stacked panels in either
    # orientation.
    if transpose:
        # module → X (7 columns), condition → Y (per-row), two stacked panels
        width = max(7, 1.2 * n_mods)
        height = max(3.0, 0.7 * len(conditions) * 2 + 2.0)
    else:
        # condition → X, module → Y; two stacked panels means ~8" tall
        width = max(7, 0.9 * len(conditions))
        height = 8

    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    for ax, matrix, title in [
        (axes[0], sim_matrix, "US-Centric/Western"),
        (axes[1], dist_matrix, "US-Distant/Non-Western"),
    ]:
        if transpose:
            ax.set_xticks(range(n_mods))
            ax.set_xticklabels(MODULE_ORDER, fontsize=9, rotation=30, ha="right")
            ax.set_yticks(range(len(conditions)))
            ax.set_yticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
            ax.set_xlabel("Module")
            ax.set_ylabel("Condition")
            display = matrix.T
        else:
            ax.set_yticks(range(n_mods))
            ax.set_yticklabels(MODULE_ORDER, fontsize=9)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=8)
            ax.set_xlabel("Condition")
            ax.set_ylabel("Module")
            display = matrix
        im = ax.imshow(display, aspect="auto", cmap="magma", origin="upper",
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="Mean per-item Attribution")
    sub = " (normad − normadcontrol)" if subtract_control else " (normad raw)"
    fig.suptitle("Per-Neuron Attribution by Group × Module" + sub, fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_group_attribution_by_module{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure_asymmetry_attribution(conditions: list[str], suffix: str,
                                  subtract_control: bool,
                                  scale: str = "robust",
                                  robust_pct: float = 95.0,
                                  symlog_linthresh: float | None = None,
                                  transpose: bool = False):
    """Single-panel heatmap: layer × condition, cell = (US-similar mean) −
    (US-distant mean) of per-item attribution. Diverging colormap centered at 0.

    Red cell = neuron-level attribution at that (layer, condition) is biased
    toward western/US-similar countries. Blue = biased toward non-western.

    `scale` controls how the cell values are mapped to colors:
      - "linear":      vmin/vmax = ±max(|diff|). Faithful, but a single extreme
                       cell can wash everything else out.
      - "robust":      vmin/vmax = ±percentile(|diff|, robust_pct). Cells beyond
                       the percentile are clipped (saturated) so the bulk of the
                       data uses the full color range. Default.
      - "symlog":      symmetric-log normalization. Compresses large magnitudes
                       and expands small ones — good when one column is an order
                       of magnitude bigger than the others.
      - "column-norm": divide each column by its own max(|diff|). Each column
                       gets its own [-1, +1] range, so the SHAPE of asymmetry
                       is visible in every column regardless of magnitude.
                       Trade-off: you lose absolute-magnitude comparability
                       across columns — only useful for "where in the depth
                       profile is asymmetry concentrated?" not "which condition
                       has more asymmetry?".
    """
    iw = load_iw_coords(IW_COORDS)
    if iw is None:
        return
    country_to_cluster, _ = iw
    item_counts = per_country_item_counts()
    if not item_counts:
        print("[warn] per-country item counts unavailable; "
              "skipping asymmetry_attribution heatmap", file=sys.stderr)
        return

    n_layers = 28
    diff = np.full((n_layers, len(conditions)), np.nan)
    for j, cond in enumerate(conditions):
        sim, dist = _compute_layer_group_attribution(
            cond, country_to_cluster, item_counts, subtract_control, n_layers
        )
        diff[:, j] = sim - dist

    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        print("[warn] no finite values for asymmetry_attribution; skipping",
              file=sys.stderr)
        return

    # Pick the matrix to plot + how to map values → colors, per `scale`.
    plot_matrix = diff
    norm = None
    cbar_label = "Δ mean per-item attribution (sim − dist)"
    scale_tag = scale
    if scale == "column-norm":
        # Normalize each column by its own max(|diff|). Columns with all-NaN
        # are left as-is (still NaN). A column whose values are all 0 also stays
        # 0 (skip the division).
        plot_matrix = diff.copy()
        for j in range(diff.shape[1]):
            col = diff[:, j]
            col_finite = col[np.isfinite(col)]
            if col_finite.size == 0:
                continue
            col_amax = float(np.max(np.abs(col_finite)))
            if col_amax > 0:
                plot_matrix[:, j] = col / col_amax
        vmin, vmax = -1.0, 1.0
        cbar_label += "  (column-normalized to [-1, +1])"
    elif scale == "symlog":
        amax = float(np.max(np.abs(finite)))
        # linthresh = where the log scale transitions to linear near 0. Default
        # to a fraction of the median |diff| so small-magnitude conditions
        # still get meaningful color resolution.
        if symlog_linthresh is None:
            med = float(np.median(np.abs(finite)))
            linthresh = max(med, amax * 1e-3)
        else:
            linthresh = symlog_linthresh
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-amax, vmax=amax, base=10)
        scale_tag = f"symlog (linthresh={linthresh:.2e})"
    elif scale == "robust":
        q = float(np.percentile(np.abs(finite), robust_pct))
        if q == 0:  # fall back to max if percentile is 0 (degenerate data)
            q = float(np.max(np.abs(finite))) or 1e-12
        vmin, vmax = -q, q
        scale_tag = f"robust p{robust_pct:.0f} (clipped at ±{q:.2e})"
    else:  # "linear"
        amax = float(np.max(np.abs(finite)))
        vmin, vmax = -amax, amax

    fig, ax = plt.subplots(figsize=_layer_axis_figsize(len(conditions), transpose))
    display = _setup_layer_axis(ax, plot_matrix, conditions, n_layers, transpose)
    if norm is not None:
        im = ax.imshow(display, aspect="auto", cmap="RdBu_r",
                       origin="upper", norm=norm)
    else:
        im = ax.imshow(display, aspect="auto", cmap="RdBu_r",
                       origin="upper", vmin=vmin, vmax=vmax)
    sub = " (normad − normadcontrol)" if subtract_control else " (normad raw)"
    ax.set_title(f"Attribution asymmetry: US-similar − US-distant{sub}\n"
                 f"scale: {scale_tag}", fontsize=10)
    plt.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    out = FIGURES_DIR / f"heatmap_asymmetry_attribution{suffix}.pdf"
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
             "E.g., --exclude dpo to skip the DPO column.",
    )
    parser.add_argument(
        "--figures", nargs="+",
        choices=["layer_count", "layer_attribution", "module_distribution",
                 "cluster_activation", "group_attribution",
                 "group_attribution_per_condition",
                 "group_attribution_by_module",
                 "group_attribution_per_condition_by_module",
                 "asymmetry_attribution", "all"],
        default=["all"],
        help="Which heatmaps to generate. 'all' covers the original four "
             "heatmaps. The group_* / asymmetry_* heatmaps are opt-in because "
             "they depend on the same US-similar/US-distant split as the bar plots.",
    )
    parser.add_argument(
        "--subtract-control", action="store_true",
        help="For group_attribution / asymmetry_attribution: subtract the "
             "per-country normadcontrol attribution from the normad "
             "attribution before averaging (per-country analogue of the "
             "scalar attribute_score subtraction in decide_culture_neurons).",
    )
    parser.add_argument(
        "--asymmetry-scale",
        choices=["linear", "robust", "symlog", "column-norm"],
        default="robust",
        help="Color-scale mapping for the asymmetry heatmap. 'linear' is the "
             "naive ±max(|diff|), which can wash out smaller-magnitude columns "
             "when one column dominates. 'robust' (default) clips to a "
             "percentile of |diff|. 'symlog' compresses large magnitudes. "
             "'column-norm' rescales each column to [-1, +1] independently "
             "(use for shape-comparison only — kills cross-column magnitude "
             "comparability).",
    )
    parser.add_argument(
        "--asymmetry-robust-pct", type=float, default=95.0,
        help="Percentile used by --asymmetry-scale=robust. Default 95.",
    )
    parser.add_argument(
        "--asymmetry-symlog-linthresh", type=float, default=None,
        help="Linear-region threshold for --asymmetry-scale=symlog. Default: "
             "median |diff| (auto).",
    )
    parser.add_argument(
        "--transpose", action="store_true",
        help="Flip layer ↔ condition axes for layer_count, layer_attribution, "
             "group_attribution, and asymmetry_attribution. Useful when you "
             "have many layers (28) and few conditions (~3-4): the figure "
             "becomes a wide banner (layer on X, condition on Y) instead of a "
             "tall narrow column.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Explicit ordered list of conditions, overriding --setup/--exclude. "
             "Use this to render a heatmap with a specific small subset of "
             "conditions, e.g. --conditions base sft sftdpo.",
    )
    parser.add_argument(
        "--out-suffix", default=None,
        help="Override the auto-generated filename suffix. Use this when "
             "producing multiple variants of the same figure type (e.g. two "
             "different --conditions sets) so they don't overwrite.",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.conditions:
        conditions = args.conditions
    else:
        conditions = [c for c in SETUP_CONDITIONS[args.setup] if c not in args.exclude]

    # Suffix encodes both setup and exclusions so figures don't overwrite.
    if args.out_suffix is not None:
        suffix = args.out_suffix
    elif args.conditions:
        # No human-readable preset name; derive a short hash-y tag from the list
        suffix = "_" + "_".join(args.conditions)
    else:
        suffix = "" if (args.setup == "all" and not args.exclude) else f"_{args.setup}"
        if args.exclude:
            suffix += "_no_" + "_".join(sorted(args.exclude))

    todo = ({"layer_count", "layer_attribution", "module_distribution", "cluster_activation"}
            if "all" in args.figures else set(args.figures))

    if "layer_count" in todo:
        figure_layer_count(conditions, suffix, transpose=args.transpose)
    if "layer_attribution" in todo:
        figure_layer_attribution(conditions, suffix, transpose=args.transpose)
    if "module_distribution" in todo:    figure_module_distribution(conditions, suffix)
    if "cluster_activation" in todo:     figure_cluster_activation(conditions, suffix)
    if "group_attribution" in todo:
        figure_group_attribution(conditions, suffix, args.subtract_control,
                                 transpose=args.transpose)
    if "group_attribution_per_condition" in todo:
        figure_group_attribution_per_condition(
            conditions, suffix, args.subtract_control
        )
    if "group_attribution_by_module" in todo:
        figure_group_attribution_by_module(
            conditions, suffix, args.subtract_control,
            transpose=args.transpose,
        )
    if "group_attribution_per_condition_by_module" in todo:
        figure_group_attribution_per_condition_by_module(
            conditions, suffix, args.subtract_control
        )
    if "asymmetry_attribution" in todo:
        figure_asymmetry_attribution(
            conditions, suffix, args.subtract_control,
            scale=args.asymmetry_scale,
            robust_pct=args.asymmetry_robust_pct,
            symlog_linthresh=args.asymmetry_symlog_linthresh,
            transpose=args.transpose,
        )


if __name__ == "__main__":
    main()
