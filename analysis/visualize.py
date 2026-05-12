"""Generate the three paper figures.

Figure 1 — Bar chart: NormAd accuracy by culture group (Western, Non-Western) across
           the four conditions; shows the widening gap.
Figure 2 — Heatmap: NormAd-identified culture-neuron mean attribution score, by
           layer x condition; the "fading" visualization showing alignment suppression.
Figure 3 — Stacked bar: NormAd-only vs. Shared vs. BLEnD-only neurons per condition;
           validates that the NormAd extension picks up overlapping-but-distinct neurons.

Reads:
  outputs/behavioral/normad_{cond}.json     (Figure 1)
  outputs/neurons/{cond}/all_neurons_normad_max.json   (Figure 2)
  outputs/neurons/attribution_summary.json  (Figure 3)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
NEURONS_DIR = PROJECT_ROOT / "outputs" / "neurons"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

CONDITIONS = ["base", "sft", "dpo", "instruct"]
COND_LABELS = {"base": "C1: Base", "sft": "C2: SFT", "dpo": "C3: DPO", "instruct": "C4: Instruct"}


def figure1_accuracy_bars():
    western, non_western = [], []
    for cond in CONDITIONS:
        path = BEHAVIORAL_DIR / f"normad_{cond}.json"
        if not path.exists():
            western.append(np.nan); non_western.append(np.nan); continue
        d = json.loads(path.read_text())
        g = d.get("accuracy_by_group", {})
        western.append(g.get("Western") or np.nan)
        non_western.append(g.get("Non-Western") or np.nan)

    x = np.arange(len(CONDITIONS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, western, w, label="Western", color="#4477AA")
    ax.bar(x + w / 2, non_western, w, label="Non-Western", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_ylabel("NormAd accuracy")
    ax.set_title("Figure 1. NormAd accuracy by culture group across conditions")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_normad_by_group.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure2_neuron_heatmap():
    """Per-layer mean attribution score across conditions (NormAd-identified neurons)."""
    rows = []
    max_layer = 0
    for cond in CONDITIONS:
        path = NEURONS_DIR / cond / "all_neurons_normad_max.json"
        if not path.exists():
            rows.append(None); continue
        data = json.loads(path.read_text())
        by_layer = defaultdict(list)
        for n in data["top_neurons"]:
            by_layer[n["layer_idx"]].append(n["attribute_score"])
        max_layer = max(max_layer, max(by_layer.keys(), default=-1))
        rows.append(by_layer)

    if max_layer < 0:
        print("[fig2] no neuron data; skipping")
        return
    n_layers = max_layer + 1
    mat = np.full((len(CONDITIONS), n_layers), np.nan)
    for i, row in enumerate(rows):
        if row is None:
            continue
        for L, scores in row.items():
            mat[i, L] = float(np.mean(scores))

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS])
    ax.set_xlabel("Layer")
    ax.set_title("Figure 2. NormAd-identified culture-neuron attribution by layer")
    plt.colorbar(im, ax=ax, label="Mean (NormAd − NormAdctrl) attribution score")
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_neuron_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def figure3_overlap():
    """Per-condition stacked bar of NormAd-only / shared / BLEnD-only neurons."""
    summary_path = NEURONS_DIR / "attribution_summary.json"
    if not summary_path.exists():
        print("[fig3] run analysis/neuron_attribution.py first")
        return
    summary = json.loads(summary_path.read_text())
    cross = summary.get("cross_source_overlap", {})
    if not cross:
        print("[fig3] no cross-source overlap data; need both BLEnD and NormAd runs")
        return

    conds = [c for c in CONDITIONS if c in cross]
    normad_only = [cross[c]["normad_only"] for c in conds]
    shared = [cross[c]["shared"] for c in conds]
    blend_only = [cross[c]["blend_only"] for c in conds]

    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, normad_only, label="NormAd only", color="#CC6677")
    ax.bar(x, shared, bottom=normad_only, label="Shared", color="#999933")
    ax.bar(x, blend_only,
           bottom=[a + b for a, b in zip(normad_only, shared)],
           label="BLEnD only", color="#4477AA")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in conds])
    ax.set_ylabel("# culture neurons")
    ax.set_title("Figure 3. NormAd-identified vs. BLEnD-identified neuron overlap")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_overlap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", nargs="+", choices=["1", "2", "3", "all"], default=["all"])
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    todo = {"1", "2", "3"} if "all" in args.figures else set(args.figures)
    if "1" in todo: figure1_accuracy_bars()
    if "2" in todo: figure2_neuron_heatmap()
    if "3" in todo: figure3_overlap()


if __name__ == "__main__":
    main()
