"""Generate 12 combined pipeline figures: 3 models × 2 benchmarks × 2 metrics.

Each figure has two subplots:
  top:    binary (US-similar vs US-distant)
  bottom: I-W clusters

Models:
  llama  — base, sft, dpo, sftdpo  (Llama 3.1 8B)
  gemma  — base, sft, dpo, sftdpo  (Gemma4)
  tulu   — base, tulu3_sft, tulu3_dpo

Benchmarks:
  normad — reads *_nfs_mpw_usprobe.json
  blend  — reads *_nfs_usprobe.json (no mpw)

Metrics:
  accuracy       — fraction correct
  us_default_rate — among errors, fraction matching US prediction
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL   = PROJECT_ROOT / "outputs" / "behavioral"
IW_COORDS    = PROJECT_ROOT / "data" / "iw_coordinates.csv"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
US_SIMILAR_CLUSTERS = {"EnglishSpeaking", "ProtestantEurope"}
COLOR_US_SIMILAR    = "#4477AA"
COLOR_US_DISTANT    = "#CC6677"
PALETTE_OTHER = ["#CC6677","#117733","#DDCC77","#882255","#88CCEE","#999933","#AA4499","#332288"]
MARKERS = ["o","s","^","D","v","P","X","*","h","<"]

COUNTRY_ALIASES = {
    "uk":               "united_kingdom",
    "us":               "united_states_of_america",
    "northern_nigeria": "nigeria",
    "assam":            "india",
    "west_java":        "indonesia",
}

# ── model configs ─────────────────────────────────────────────────────────────
MODELS = {
    "llama": dict(
        conditions=["base", "sft", "dpo", "sftdpo"],
        labels={"base": "Base", "sft": "SFT", "dpo": "DPO", "sftdpo": "SFT+DPO"},
        size="8b",
        title="Llama 3.1 8B",
    ),
    "gemma": dict(
        conditions=["base", "sft", "dpo", "sftdpo"],
        labels={"base": "Base", "sft": "SFT", "dpo": "DPO", "sftdpo": "SFT+DPO"},
        size="gemma4",
        title="Gemma4",
    ),
    "tulu": dict(
        conditions=["base", "tulu3_sft", "tulu3_dpo"],
        labels={"base": "Base", "tulu3_sft": "Tulu3-SFT", "tulu3_dpo": "Tulu3-DPO"},
        size="8b",
        title="Tulu3 (8B)",
    ),
}

BENCHMARKS = {
    "normad": dict(prefix="normad", file_suffix="_nfs_mpw_usprobe", label="NormAd"),
    "blend":  dict(prefix="blend",  file_suffix="_nfs_usprobe",     label="BLEnD"),
}

METRICS = {
    "accuracy":        "Accuracy",
    "us_default_rate": "US-Default Rate Among Errors",
}


# ── data loading ──────────────────────────────────────────────────────────────
def load_iw_data():
    country_to_cluster: dict[str, str] = {}
    cluster_dists: dict[str, list[float]] = defaultdict(list)
    with open(IW_COORDS) as f:
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


def resolve_cluster(country, country_to_cluster):
    normed = COUNTRY_ALIASES.get(country.lower(), country.lower())
    return country_to_cluster.get(country) or country_to_cluster.get(normed)


def compute_groups(cond, size, benchmark_cfg, metric, country_to_cluster, mode):
    sfx = ("" if size == "3b" else f"_{size}") + benchmark_cfg["file_suffix"]
    path = BEHAVIORAL / f"{benchmark_cfg['prefix']}_{cond}{sfx}.json"
    if not path.exists():
        return {}

    preds = json.loads(path.read_text()).get("predictions", [])
    if metric == "accuracy":
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p in preds:
            cluster = resolve_cluster(p.get("country", ""), country_to_cluster)
            if cluster is None:
                continue
            group = "US-similar" if cluster in US_SIMILAR_CLUSTERS else (
                "US-distant" if mode == "binary" else cluster)
            counts[group][0] += int(p["gold"] == p["pred"])
            counts[group][1] += 1
        return {g: c/t for g, (c, t) in counts.items() if t > 0}
    else:  # us_default_rate
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p in preds:
            if p.get("country") == "US":
                continue
            us_pred = p.get("us_pred")
            if us_pred is None or p["pred"] == p["gold"]:
                continue
            cluster = resolve_cluster(p.get("country", ""), country_to_cluster)
            if cluster is None:
                continue
            group = "US-similar" if cluster in US_SIMILAR_CLUSTERS else (
                "US-distant" if mode == "binary" else cluster)
            counts[group][1] += 1
            if p["pred"] == us_pred:
                counts[group][0] += 1
        return {g: m/t for g, (m, t) in counts.items() if t > 0}


# ── plotting ──────────────────────────────────────────────────────────────────
def order_groups(keys, cluster_mean_dist, mode):
    if mode == "binary":
        order = []
        if "US-similar" in keys: order.append("US-similar")
        if "US-distant" in keys: order.append("US-distant")
        return order
    rest = sorted([k for k in keys if k != "US-similar"],
                  key=lambda c: cluster_mean_dist.get(c, float("inf")))
    return (["US-similar"] if "US-similar" in keys else []) + rest


def plot_subplot(ax, conditions, labels, acc_by_group, ordered_groups,
                 cluster_mean_dist, mode, metric_label, show_values):
    x = np.arange(len(conditions))
    other_idx = 0
    for i, group in enumerate(ordered_groups):
        if group == "US-similar":
            color = COLOR_US_SIMILAR
            lw, ms = 2.6, 9
        elif mode == "binary":
            color = COLOR_US_DISTANT
            lw, ms = 2.0, 7
        else:
            color = PALETTE_OTHER[other_idx % len(PALETTE_OTHER)]
            other_idx += 1
            lw, ms = 2.0, 7

        accs = acc_by_group[group]
        if group == "US-similar":
            leg_label = "US-similar"
        elif mode == "binary":
            leg_label = "US-distant"
        else:
            d = cluster_mean_dist.get(group)
            leg_label = group + (f" (d={d:.2f})" if d else "")

        ax.plot(x, accs, marker=MARKERS[i % len(MARKERS)],
                markersize=ms, linewidth=lw, color=color, label=leg_label)

        if show_values:
            for xi, v in zip(x, accs):
                if not math.isnan(v):
                    off = (0, 10) if group == "US-similar" else (0, -15)
                    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                                xytext=off, ha="center", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in conditions], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.4, len(conditions) - 0.6)

    finite = [v for vals in acc_by_group.values() for v in vals if not math.isnan(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        span = max(hi - lo, 0.02)
        ax.set_ylim(lo - span * 0.2, hi + span * 0.2)

    if mode == "binary":
        ax.legend(loc="best", fontsize=9, frameon=True)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)


def make_figure(model_key, benchmark_key, metric_key, country_to_cluster, cluster_mean_dist):
    model_cfg     = MODELS[model_key]
    benchmark_cfg = BENCHMARKS[benchmark_key]
    metric_label  = METRICS[metric_key]
    conditions    = model_cfg["conditions"]
    labels        = model_cfg["labels"]
    size          = model_cfg["size"]

    fig, axes = plt.subplots(2, 1, figsize=(max(7, 1.8 * len(conditions)) + 2, 9))
    fig.suptitle(
        f"{model_cfg['title']} — {benchmark_cfg['label']} — {metric_label}",
        fontsize=12, fontweight="bold"
    )

    for ax, mode in zip(axes, ["binary", "clusters"]):
        all_groups: dict[str, list[float]] = defaultdict(
            lambda: [float("nan")] * len(conditions))

        for j, cond in enumerate(conditions):
            groups = compute_groups(cond, size, benchmark_cfg, metric_key,
                                    country_to_cluster, mode)
            for g, v in groups.items():
                all_groups[g][j] = v

        ordered = order_groups(set(all_groups.keys()), cluster_mean_dist, mode)
        show_values = (mode == "binary")
        ax.set_title("Binary (US-similar vs US-distant)" if mode == "binary"
                     else "By I-W Cluster", fontsize=10)
        plot_subplot(ax, conditions, labels, all_groups, ordered,
                     cluster_mean_dist, mode, metric_label, show_values)

    fig.tight_layout()
    out = FIGURES_DIR / f"pipeline_{model_key}_{benchmark_key}_{metric_key}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    country_to_cluster, cluster_mean_dist = load_iw_data()
    for model_key in MODELS:
        for benchmark_key in BENCHMARKS:
            for metric_key in METRICS:
                make_figure(model_key, benchmark_key, metric_key,
                            country_to_cluster, cluster_mean_dist)


if __name__ == "__main__":
    main()
