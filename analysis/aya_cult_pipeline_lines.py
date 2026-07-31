"""Line graphs comparing aya cult vs nocult conditions across the alignment pipeline.

X axis: Base → SFT → SFT+DPO
Blue lines: Western (US-centric) — solid=cult, dotted=nocult
Red lines:  Non-Western           — solid=cult, dotted=nocult

Produces (e.g. for --model-size 8b):
  outputs/figures/aya_cult_8b_normad_accuracy.pdf
  outputs/figures/aya_cult_8b_normad_us_default_rate.pdf
  outputs/figures/aya_cult_8b_blend_accuracy.pdf
  outputs/figures/aya_cult_8b_blend_us_default_rate.pdf
  outputs/figures/aya_cult_8b_boolq_accuracy.pdf
  outputs/figures/aya_cult_8b_csqa_accuracy.pdf
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

COLOR_WESTERN     = "#4477AA"  # blue
COLOR_NONWESTERN  = "#CC6677"  # red

X_LABELS = ["Base", "SFT", "SFT+DPO"]
CULT_CONDITIONS   = ["base", "sft_aya_cult",   "sftdpo_aya_cult"]
NOCULT_CONDITIONS = ["base", "sft_aya_nocult", "sftdpo_aya_nocult"]


def load_normad_preds(cond: str, model_size: str) -> list[dict]:
    path = BEHAVIORAL_DIR / f"normad_{cond}_{model_size}_nfs_mpw_usprobe_batch_calibrated.json"
    return json.loads(path.read_text())["predictions"]


def load_blend_preds(cond: str, model_size: str) -> list[dict]:
    path = BEHAVIORAL_DIR / f"blend_{cond}_{model_size}_nfs_usprobe.json"
    return json.loads(path.read_text())["predictions"]


def load_nlu_preds(dataset: str, cond: str, model_size: str) -> list[dict]:
    path = BEHAVIORAL_DIR / f"nlu_{dataset}_{cond}_{model_size}_nfs.json"
    return json.loads(path.read_text())["predictions"]


def accuracy_overall(preds: list[dict]) -> float:
    return sum(1 for p in preds if p["pred"] == p["gold"]) / len(preds) if preds else float("nan")


def accuracy_by_group(preds: list[dict], yn_only: bool = True) -> dict[str, float]:
    groups = {}
    for group in ("Western", "Non-Western"):
        subset = [p for p in preds if p.get("group") == group]
        if yn_only:
            subset = [p for p in subset if p["gold"] in ("yes", "no")]
        groups[group] = sum(1 for p in subset if p["pred"] == p["gold"]) / len(subset) if subset else float("nan")
    return groups


def us_default_overall(preds: list[dict], yn_only: bool = True) -> float:
    errors = [p for p in preds
              if p.get("country") != "US"
              and p.get("us_pred") is not None
              and p["pred"] != p["gold"]]
    if yn_only:
        errors = [p for p in errors if p["gold"] in ("yes", "no")]
    return sum(1 for p in errors if p["pred"] == p["us_pred"]) / len(errors) if errors else float("nan")


def make_figure_overall(cult_vals: list[float], nocult_vals: list[float],
                        ylabel: str, out_path: Path, y_range: float | None = None) -> None:
    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(x, cult_vals,   marker="o", markersize=9, linewidth=2.6,
            color=COLOR_WESTERN, linestyle="-",  label="Cultural Data")
    ax.plot(x, nocult_vals, marker="o", markersize=7, linewidth=2.0,
            color=COLOR_WESTERN, linestyle="--", label="Non-Cultural Data")

    for xi, (cv, nv) in enumerate(zip(cult_vals, nocult_vals)):
        if not math.isnan(cv):
            ax.annotate(f"{cv:.3f}", (xi, cv), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=COLOR_WESTERN)
        if not math.isnan(nv):
            ax.annotate(f"{nv:.3f}", (xi, nv), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=9, color=COLOR_WESTERN)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOR_WESTERN, linestyle="-",  linewidth=2.6, marker="o", markersize=8, label="— Cultural Data"),
        Line2D([0], [0], color=COLOR_WESTERN, linestyle="--", linewidth=2.0, marker="o", markersize=8, label="- - Non-Cultural Data"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=10, frameon=True)

    finite = [v for v in cult_vals + nocult_vals if not math.isnan(v)]
    if finite:
        mid = (min(finite) + max(finite)) / 2
        half = (y_range if y_range else max(max(finite) - min(finite), 0.02) * 1.4) / 2
        ax.set_ylim(mid - half, mid + half)
    ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_figure(
    cult_western: list[float], cult_nonwestern: list[float],
    nocult_western: list[float], nocult_nonwestern: list[float],
    ylabel: str, out_path: Path, y_range: float | None = None,
) -> None:
    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    lines = [
        (cult_western,      COLOR_WESTERN,    "-",  "o", 9,  2.6, "Western (Cultural Data)"),
        (nocult_western,    COLOR_WESTERN,    "--", "o", 7,  2.0, "Western (Non-Cultural Data)"),
        (cult_nonwestern,   COLOR_NONWESTERN, "-",  "s", 9,  2.6, "Non-Western (Cultural Data)"),
        (nocult_nonwestern, COLOR_NONWESTERN, "--", "s", 7,  2.0, "Non-Western (Non-Cultural Data)"),
    ]

    for vals, color, ls, marker, ms, lw, label in lines:
        ax.plot(x, vals, marker=marker, markersize=ms, linewidth=lw,
                color=color, linestyle=ls, label=label)
        for xi, v in enumerate(vals):
            if not math.isnan(v):
                offset = (0, 10) if ls == "-" else (0, -15)
                ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                            xytext=offset, ha="center", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # Two-part legend: color entries + linestyle entries
    from matplotlib.lines import Line2D
    color_handles = [
        Line2D([0], [0], color=COLOR_WESTERN,    marker="o", markersize=8, linewidth=2, label="Western"),
        Line2D([0], [0], color=COLOR_NONWESTERN, marker="s", markersize=8, linewidth=2, label="Non-Western"),
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="Cultural Data"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Non-Cultural Data"),
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper right", fontsize=10, frameon=True, title="Group")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=10, frameon=True, title="Training Data")

    all_vals = cult_western + cult_nonwestern + nocult_western + nocult_nonwestern
    finite = [v for v in all_vals if not math.isnan(v)]
    if finite:
        mid = (min(finite) + max(finite)) / 2
        half = (y_range if y_range else max(max(finite) - min(finite), 0.02) * 1.4) / 2
        ax.set_ylim(mid - half, mid + half)
    ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b", choices=["8b", "gemma4", "qwen35"])
    args = parser.parse_args()
    ms = args.model_size
    pfx = f"aya_cult_{ms}"

    # --- NormAd ---
    cult_preds   = [load_normad_preds(c, ms) for c in CULT_CONDITIONS]
    nocult_preds = [load_normad_preds(c, ms) for c in NOCULT_CONDITIONS]

    cult_acc   = [accuracy_by_group(p, yn_only=True) for p in cult_preds]
    nocult_acc = [accuracy_by_group(p, yn_only=True) for p in nocult_preds]

    make_figure(
        cult_western      = [d["Western"]     for d in cult_acc],
        cult_nonwestern   = [d["Non-Western"] for d in cult_acc],
        nocult_western    = [d["Western"]     for d in nocult_acc],
        nocult_nonwestern = [d["Non-Western"] for d in nocult_acc],
        ylabel="NormAd Accuracy",
        out_path=FIGURES_DIR / f"{pfx}_normad_accuracy.pdf",
        y_range=0.30,
    )

    make_figure_overall(
        cult_vals   = [us_default_overall(p, yn_only=True) for p in cult_preds],
        nocult_vals = [us_default_overall(p, yn_only=True) for p in nocult_preds],
        ylabel="NormAd US-Default Rate Among Errors",
        out_path=FIGURES_DIR / f"{pfx}_normad_us_default_rate.pdf",
        y_range=0.30,
    )

    # --- BLEnD ---
    cult_blend   = [load_blend_preds(c, ms) for c in CULT_CONDITIONS]
    nocult_blend = [load_blend_preds(c, ms) for c in NOCULT_CONDITIONS]

    cult_blend_acc   = [accuracy_by_group(p, yn_only=False) for p in cult_blend]
    nocult_blend_acc = [accuracy_by_group(p, yn_only=False) for p in nocult_blend]

    make_figure(
        cult_western      = [d["Western"]     for d in cult_blend_acc],
        cult_nonwestern   = [d["Non-Western"] for d in cult_blend_acc],
        nocult_western    = [d["Western"]     for d in nocult_blend_acc],
        nocult_nonwestern = [d["Non-Western"] for d in nocult_blend_acc],
        ylabel="BLEnD Accuracy",
        out_path=FIGURES_DIR / f"{pfx}_blend_accuracy.pdf",
        y_range=0.30,
    )

    make_figure_overall(
        cult_vals   = [us_default_overall(p, yn_only=False) for p in cult_blend],
        nocult_vals = [us_default_overall(p, yn_only=False) for p in nocult_blend],
        ylabel="BLEnD US-Default Rate Among Errors",
        out_path=FIGURES_DIR / f"{pfx}_blend_us_default_rate.pdf",
        y_range=0.30,
    )

    # --- NLU: BoolQ and CSQA (overall accuracy only, no group split) ---
    for dataset, label in [("boolq", "BoolQ"), ("csqa", "CSQA")]:
        cult_nlu   = [load_nlu_preds(dataset, c, ms) for c in CULT_CONDITIONS]
        nocult_nlu = [load_nlu_preds(dataset, c, ms) for c in NOCULT_CONDITIONS]
        make_figure_overall(
            cult_vals   = [accuracy_overall(p) for p in cult_nlu],
            nocult_vals = [accuracy_overall(p) for p in nocult_nlu],
            ylabel=f"{label} Accuracy",
            out_path=FIGURES_DIR / f"{pfx}_{dataset}_accuracy.pdf",
            y_range=0.30,
        )


if __name__ == "__main__":
    main()
