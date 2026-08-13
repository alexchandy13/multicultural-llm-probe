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


def us_default_all(preds: list[dict], yn_only: bool = True) -> float:
    """US-default rate over all non-US predictions (not just errors)."""
    subset = [p for p in preds
              if p.get("country") != "US"
              and p.get("us_pred") is not None]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    return sum(1 for p in subset if p["pred"] == p["us_pred"]) / len(subset) if subset else float("nan")


def make_figure_overall(cult_vals: list[float], nocult_vals: list[float],
                        ylabel: str, out_path: Path, y_range: float | None = None,
                        ref_val: float | None = None) -> None:
    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(x, cult_vals,   marker="o", markersize=9, linewidth=2.6,
            color=COLOR_WESTERN, linestyle="-",  label="Cultural Data")
    ax.plot(x, nocult_vals, marker="o", markersize=7, linewidth=2.0,
            color=COLOR_WESTERN, linestyle="--", label="Non-Cultural Data")

    if ref_val is not None and not math.isnan(ref_val):
        ax.axhline(ref_val, color="gray", linestyle=":", linewidth=1.6, label="Instruct")

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOR_WESTERN, linestyle="-",  linewidth=2.6, marker="o", markersize=8, label="Cultural Data"),
        Line2D([0], [0], color=COLOR_WESTERN, linestyle="--", linewidth=2.0, marker="o", markersize=8, label="Non-Cultural Data"),
    ]
    if ref_val is not None and not math.isnan(ref_val):
        handles.append(Line2D([0], [0], color="gray", linestyle=":", linewidth=1.6, label="Instruct"))
    ax.legend(handles=handles, loc="best", fontsize=10, frameon=True)

    finite = [v for v in cult_vals + nocult_vals if not math.isnan(v)]
    if ref_val is not None and not math.isnan(ref_val):
        finite.append(ref_val)
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
    ref_western: float | None = None, ref_nonwestern: float | None = None,
) -> None:
    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    lines = [
        (cult_western,      COLOR_WESTERN,    "-",  "o", 9,  2.6),
        (nocult_western,    COLOR_WESTERN,    "--", "o", 7,  2.0),
        (cult_nonwestern,   COLOR_NONWESTERN, "-",  "s", 9,  2.6),
        (nocult_nonwestern, COLOR_NONWESTERN, "--", "s", 7,  2.0),
    ]

    for vals, color, ls, marker, ms, lw in lines:
        ax.plot(x, vals, marker=marker, markersize=ms, linewidth=lw,
                color=color, linestyle=ls)

    if ref_western is not None and not math.isnan(ref_western):
        ax.axhline(ref_western, color=COLOR_WESTERN, linestyle=":", linewidth=1.4, alpha=0.7)
    if ref_nonwestern is not None and not math.isnan(ref_nonwestern):
        ax.axhline(ref_nonwestern, color=COLOR_NONWESTERN, linestyle=":", linewidth=1.4, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    color_handles = [
        Line2D([0], [0], color=COLOR_WESTERN,    marker="o", markersize=8, linewidth=2, label="Western"),
        Line2D([0], [0], color=COLOR_NONWESTERN, marker="s", markersize=8, linewidth=2, label="Non-Western"),
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="Cultural Data"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Non-Cultural Data"),
    ]
    has_ref = (ref_western is not None and not math.isnan(ref_western)) or \
              (ref_nonwestern is not None and not math.isnan(ref_nonwestern))
    if has_ref:
        style_handles.append(Line2D([0], [0], color="gray", linestyle=":", linewidth=1.4, label="Instruct"))
    leg1 = ax.legend(handles=color_handles, loc="upper right", fontsize=10, frameon=True, title="Group")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="lower left", fontsize=10, frameon=True, title="Training Data")

    all_vals = cult_western + cult_nonwestern + nocult_western + nocult_nonwestern
    finite = [v for v in all_vals if not math.isnan(v)]
    if ref_western   is not None and not math.isnan(ref_western):   finite.append(ref_western)
    if ref_nonwestern is not None and not math.isnan(ref_nonwestern): finite.append(ref_nonwestern)
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


def _try_load_normad(cond: str, model_size: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"normad_{cond}_{model_size}_nfs_mpw_usprobe_batch_calibrated.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def _try_load_blend(cond: str, model_size: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"blend_{cond}_{model_size}_nfs_usprobe.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def _try_load_nlu(dataset: str, cond: str, model_size: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"nlu_{dataset}_{cond}_{model_size}_nfs.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b", choices=["8b", "gemma4", "qwen35"])
    parser.add_argument("--instruct-size", default=None,
                        help="Model size label for instruct reference lines "
                             "(e.g. 8b_instruct, gemma4_instruct). "
                             "If not set, auto-detected from --model-size.")
    args = parser.parse_args()
    ms = args.model_size
    pfx = f"aya_cult_{ms}"

    # Auto-detect instruct variant if not specified
    if args.instruct_size:
        instruct_ms = args.instruct_size
    elif ms == "8b":
        instruct_ms = "8b_instruct"
    else:
        instruct_ms = None

    # --- NormAd ---
    cult_preds   = [load_normad_preds(c, ms) for c in CULT_CONDITIONS]
    nocult_preds = [load_normad_preds(c, ms) for c in NOCULT_CONDITIONS]

    cult_acc   = [accuracy_by_group(p, yn_only=True) for p in cult_preds]
    nocult_acc = [accuracy_by_group(p, yn_only=True) for p in nocult_preds]

    # Instruct reference for NormAd
    ref_normad_w = ref_normad_nw = None
    ref_normad_us_err = ref_normad_us_all = None
    if instruct_ms:
        inst_normad = _try_load_normad("base", instruct_ms)
        if inst_normad:
            g = accuracy_by_group(inst_normad, yn_only=True)
            ref_normad_w  = g["Western"]
            ref_normad_nw = g["Non-Western"]
            ref_normad_us_err = us_default_overall(inst_normad, yn_only=True)
            ref_normad_us_all = us_default_all(inst_normad, yn_only=True)
            print(f"Instruct NormAd ref: W={ref_normad_w:.3f} NW={ref_normad_nw:.3f}")

    make_figure(
        cult_western      = [d["Western"]     for d in cult_acc],
        cult_nonwestern   = [d["Non-Western"] for d in cult_acc],
        nocult_western    = [d["Western"]     for d in nocult_acc],
        nocult_nonwestern = [d["Non-Western"] for d in nocult_acc],
        ylabel="NormAd Accuracy",
        out_path=FIGURES_DIR / f"{pfx}_normad_accuracy.pdf",
        y_range=0.30,
        ref_western=ref_normad_w,
        ref_nonwestern=ref_normad_nw,
    )

    make_figure_overall(
        cult_vals   = [us_default_overall(p, yn_only=True) for p in cult_preds],
        nocult_vals = [us_default_overall(p, yn_only=True) for p in nocult_preds],
        ylabel="NormAd US-Default Rate Among Errors",
        out_path=FIGURES_DIR / f"{pfx}_normad_us_default_rate.pdf",
        y_range=0.30,
        ref_val=ref_normad_us_err,
    )

    make_figure_overall(
        cult_vals   = [us_default_all(p, yn_only=True) for p in cult_preds],
        nocult_vals = [us_default_all(p, yn_only=True) for p in nocult_preds],
        ylabel="NormAd Overall US-Default Rate",
        out_path=FIGURES_DIR / f"{pfx}_normad_us_default_rate_all.pdf",
        y_range=0.30,
        ref_val=ref_normad_us_all,
    )

    # --- BLEnD ---
    cult_blend   = [load_blend_preds(c, ms) for c in CULT_CONDITIONS]
    nocult_blend = [load_blend_preds(c, ms) for c in NOCULT_CONDITIONS]

    cult_blend_acc   = [accuracy_by_group(p, yn_only=False) for p in cult_blend]
    nocult_blend_acc = [accuracy_by_group(p, yn_only=False) for p in nocult_blend]

    # Instruct reference for BLEnD
    ref_blend_w = ref_blend_nw = None
    ref_blend_us_err = ref_blend_us_all = None
    if instruct_ms:
        inst_blend = _try_load_blend("base", instruct_ms)
        if inst_blend:
            g = accuracy_by_group(inst_blend, yn_only=False)
            ref_blend_w  = g["Western"]
            ref_blend_nw = g["Non-Western"]
            ref_blend_us_err = us_default_overall(inst_blend, yn_only=False)
            ref_blend_us_all = us_default_all(inst_blend, yn_only=False)
            print(f"Instruct BLEnD ref: W={ref_blend_w:.3f} NW={ref_blend_nw:.3f}")

    make_figure(
        cult_western      = [d["Western"]     for d in cult_blend_acc],
        cult_nonwestern   = [d["Non-Western"] for d in cult_blend_acc],
        nocult_western    = [d["Western"]     for d in nocult_blend_acc],
        nocult_nonwestern = [d["Non-Western"] for d in nocult_blend_acc],
        ylabel="BLEnD Accuracy",
        out_path=FIGURES_DIR / f"{pfx}_blend_accuracy.pdf",
        y_range=0.30,
        ref_western=ref_blend_w,
        ref_nonwestern=ref_blend_nw,
    )

    make_figure_overall(
        cult_vals   = [us_default_overall(p, yn_only=False) for p in cult_blend],
        nocult_vals = [us_default_overall(p, yn_only=False) for p in nocult_blend],
        ylabel="BLEnD US-Default Rate Among Errors",
        out_path=FIGURES_DIR / f"{pfx}_blend_us_default_rate.pdf",
        y_range=0.30,
        ref_val=ref_blend_us_err,
    )

    make_figure_overall(
        cult_vals   = [us_default_all(p, yn_only=False) for p in cult_blend],
        nocult_vals = [us_default_all(p, yn_only=False) for p in nocult_blend],
        ylabel="BLEnD Overall US-Default Rate",
        out_path=FIGURES_DIR / f"{pfx}_blend_us_default_rate_all.pdf",
        y_range=0.30,
        ref_val=ref_blend_us_all,
    )

    # --- NLU: BoolQ and CSQA (overall accuracy only, no group split) ---
    for dataset, label in [("boolq", "BoolQ"), ("csqa", "CSQA")]:
        cult_nlu   = [load_nlu_preds(dataset, c, ms) for c in CULT_CONDITIONS]
        nocult_nlu = [load_nlu_preds(dataset, c, ms) for c in NOCULT_CONDITIONS]
        ref_nlu = None
        if instruct_ms:
            inst_nlu = _try_load_nlu(dataset, "base", instruct_ms)
            if inst_nlu:
                ref_nlu = accuracy_overall(inst_nlu)
                print(f"Instruct {label} ref: {ref_nlu:.3f}")
        make_figure_overall(
            cult_vals   = [accuracy_overall(p) for p in cult_nlu],
            nocult_vals = [accuracy_overall(p) for p in nocult_nlu],
            ylabel=f"{label} Accuracy",
            out_path=FIGURES_DIR / f"{pfx}_{dataset}_accuracy.pdf",
            y_range=0.30,
            ref_val=ref_nlu,
        )


if __name__ == "__main__":
    main()
