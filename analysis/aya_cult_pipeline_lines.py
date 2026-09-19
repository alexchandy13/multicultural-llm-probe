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

COLOR_WESTERN     = "#1877F2"  # Meta blue (LLaMA model color)
COLOR_NONWESTERN  = "#CC6677"  # red (kept for backward compat)
COLOR_GEMMA4      = "#3DDC84"  # Android green (Gemma model color)
COLOR_W_GROUP     = "#7B52AB"  # purple  (Western group lines)
COLOR_NW_GROUP    = "#F28500"  # orange  (Non-Western group lines)

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


def us_default_errors_nw(preds: list[dict], yn_only: bool = True) -> float:
    """US-default rate among errors, Non-Western examples only."""
    subset = [p for p in preds
              if p.get("group") == "Non-Western"
              and p.get("country") != "US"
              and p.get("us_pred") is not None
              and p["pred"] != p["gold"]]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    return sum(1 for p in subset if p["pred"] == p["us_pred"]) / len(subset) if subset else float("nan")


def us_default_all(preds: list[dict], yn_only: bool = True) -> float:
    """US-default rate over all non-US predictions (not just errors)."""
    subset = [p for p in preds
              if p.get("country") != "US"
              and p.get("us_pred") is not None]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    return sum(1 for p in subset if p["pred"] == p["us_pred"]) / len(subset) if subset else float("nan")


def bootstrap_ci(preds: list, stat_fn, n_boot: int = 1000, ci: float = 95, seed: int = 42):
    """Return (lo, hi) bootstrap CI for stat_fn applied to preds."""
    if not preds:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(preds)
    stats = [stat_fn([preds[i] for i in rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    alpha = (100 - ci) / 2
    return float(np.percentile(stats, alpha)), float(np.percentile(stats, 100 - alpha))


def _group_ci(preds: list, group: str, yn_only: bool) -> tuple:
    """Bootstrap CI for accuracy of a specific group within preds."""
    subset = [p for p in preds if p.get("group") == group]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    def _acc(ps): return sum(1 for p in ps if p["pred"] == p["gold"]) / len(ps) if ps else float("nan")
    return bootstrap_ci(subset, _acc)


def muted_color(color: str, factor: float = 0.5) -> tuple:
    """Blend a hex color toward white by `factor` (0 = original, 1 = white)."""
    import matplotlib.colors as mc
    r, g, b, _ = mc.to_rgba(color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def make_bar_figure(cult_vals: list[float], nocult_vals: list[float],
                    ylabel: str, out_path: Path, color: str = COLOR_WESTERN,
                    y_range: float = 0.30) -> None:
    """Grouped bar chart: cult = solid color, nocult = muted variant."""
    x = np.arange(len(X_LABELS))
    width = 0.35
    muted = muted_color(color)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(x - width / 2, cult_vals,   width, color=color,  label="Cultural Data",     zorder=3)
    ax.bar(x + width / 2, nocult_vals, width, color=muted,  label="Non-Cultural Data", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=10, frameon=True)

    finite = [v for v in cult_vals + nocult_vals if not math.isnan(v)]
    if finite:
        mid = (min(finite) + max(finite)) / 2
        half = y_range / 2
        ax.set_ylim(max(0, mid - half), mid + half)
    ax.set_xlim(-0.5, len(X_LABELS) - 0.5)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_figure_overall(cult_vals: list[float], nocult_vals: list[float],
                        ylabel: str, out_path: Path, y_range: float | None = None,
                        ref_val: float | None = None,
                        color: str = COLOR_WESTERN) -> None:
    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.plot(x, cult_vals,   marker="o", markersize=9, linewidth=2.6,
            color=color, linestyle="-",  label="Cultural Data")
    ax.plot(x, nocult_vals, marker="o", markersize=7, linewidth=2.0,
            color=color, linestyle="--", label="Non-Cultural Data")

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
        Line2D([0], [0], color=color, linestyle="-",  linewidth=2.6, marker="o", markersize=8, label="Cultural Data"),
        Line2D([0], [0], color=color, linestyle="--", linewidth=2.0, marker="o", markersize=8, label="Non-Cultural Data"),
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
        (cult_western,      COLOR_W_GROUP,  "-",  "o", 9,  2.6),
        (nocult_western,    COLOR_W_GROUP,  "--", "o", 7,  2.0),
        (cult_nonwestern,   COLOR_NW_GROUP, "-",  "s", 9,  2.6),
        (nocult_nonwestern, COLOR_NW_GROUP, "--", "s", 7,  2.0),
    ]

    for vals, color, ls, marker, ms, lw in lines:
        ax.plot(x, vals, marker=marker, markersize=ms, linewidth=lw,
                color=color, linestyle=ls)

    if ref_western is not None and not math.isnan(ref_western):
        ax.axhline(ref_western, color=COLOR_W_GROUP, linestyle=":", linewidth=1.4, alpha=0.7)
    if ref_nonwestern is not None and not math.isnan(ref_nonwestern):
        ax.axhline(ref_nonwestern, color=COLOR_NW_GROUP, linestyle=":", linewidth=1.4, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    from matplotlib.lines import Line2D
    color_handles = [
        Line2D([0], [0], color=COLOR_W_GROUP,  marker="o", markersize=8, linewidth=2, label="US-Centric"),
        Line2D([0], [0], color=COLOR_NW_GROUP, marker="s", markersize=8, linewidth=2, label="US-Distant"),
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

    instruct_ms = args.instruct_size or None
    model_color = COLOR_GEMMA4 if ms == "gemma4" else COLOR_WESTERN

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

    make_bar_figure(
        cult_vals   = [us_default_overall(p, yn_only=True) for p in cult_preds],
        nocult_vals = [us_default_overall(p, yn_only=True) for p in nocult_preds],
        ylabel="NormAd US-Default Rate Among Errors",
        out_path=FIGURES_DIR / f"{pfx}_normad_us_default_rate.pdf",
        color=model_color,
    )

    make_bar_figure(
        cult_vals   = [us_default_errors_nw(p, yn_only=True) for p in cult_preds],
        nocult_vals = [us_default_errors_nw(p, yn_only=True) for p in nocult_preds],
        ylabel="NormAd US-Default Rate Among Errors (US-Distant)",
        out_path=FIGURES_DIR / f"{pfx}_normad_us_default_rate_nw.pdf",
        color=model_color,
    )

    make_bar_figure(
        cult_vals   = [us_default_all(p, yn_only=True) for p in cult_preds],
        nocult_vals = [us_default_all(p, yn_only=True) for p in nocult_preds],
        ylabel="NormAd Overall US-Default Rate",
        out_path=FIGURES_DIR / f"{pfx}_normad_us_default_rate_all.pdf",
        color=model_color,
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


    make_bar_figure(
        cult_vals   = [us_default_overall(p, yn_only=False) for p in cult_blend],
        nocult_vals = [us_default_overall(p, yn_only=False) for p in nocult_blend],
        ylabel="BLEnD US-Default Rate Among Errors",
        out_path=FIGURES_DIR / f"{pfx}_blend_us_default_rate.pdf",
        color=model_color,
    )

    make_bar_figure(
        cult_vals   = [us_default_errors_nw(p, yn_only=False) for p in cult_blend],
        nocult_vals = [us_default_errors_nw(p, yn_only=False) for p in nocult_blend],
        ylabel="BLEnD US-Default Rate Among Errors (US-Distant)",
        out_path=FIGURES_DIR / f"{pfx}_blend_us_default_rate_nw.pdf",
        color=model_color,
    )

    make_bar_figure(
        cult_vals   = [us_default_all(p, yn_only=False) for p in cult_blend],
        nocult_vals = [us_default_all(p, yn_only=False) for p in nocult_blend],
        ylabel="BLEnD Overall US-Default Rate",
        out_path=FIGURES_DIR / f"{pfx}_blend_us_default_rate_all.pdf",
        color=model_color,
    )

    # --- NLU + BLEnD overall: combined three-panel figure ---
    panel_data = []
    for dataset, label in [("boolq", "BoolQ"), ("csqa", "CSQA")]:
        cult_nlu   = [load_nlu_preds(dataset, c, ms) for c in CULT_CONDITIONS]
        nocult_nlu = [load_nlu_preds(dataset, c, ms) for c in NOCULT_CONDITIONS]
        panel_data.append((
            label,
            [accuracy_overall(p) for p in cult_nlu],
            [accuracy_overall(p) for p in nocult_nlu],
        ))
    # BLEnD overall accuracy
    panel_data.append((
        "BLEnD",
        [accuracy_overall(p) for p in cult_blend],
        [accuracy_overall(p) for p in nocult_blend],
    ))

    x = np.arange(len(X_LABELS))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    from matplotlib.lines import Line2D

    # Shared y-axis limits across all three panels
    all_finite = [v for _, cv, nv in panel_data for v in cv + nv if not math.isnan(v)]
    shared_mid  = (min(all_finite) + max(all_finite)) / 2
    shared_half = max(max(all_finite) - min(all_finite), 0.02) * 1.4 / 2

    for ax, (label, cult_vals, nocult_vals) in zip(axes, panel_data):
        ax.plot(x, cult_vals,   marker="o", markersize=9, linewidth=2.6,
                color=model_color, linestyle="-")
        ax.plot(x, nocult_vals, marker="o", markersize=7, linewidth=2.0,
                color=model_color, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=11)
        ax.set_xlabel("Alignment Condition", fontsize=11)
        ax.set_ylabel(f"{label} Accuracy", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.set_ylim(shared_mid - shared_half, shared_mid + shared_half)
        ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    handles = [
        Line2D([0], [0], color=model_color, linestyle="-",  linewidth=2.6,
               marker="o", markersize=8, label="Cultural Data"),
        Line2D([0], [0], color=model_color, linestyle="--", linewidth=2.0,
               marker="o", markersize=8, label="Non-Cultural Data"),
    ]
    axes[2].legend(handles=handles, loc="best", fontsize=10, frameon=True)

    fig.tight_layout()
    nlu_out = FIGURES_DIR / f"{pfx}_nlu_accuracy.pdf"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(nlu_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {nlu_out}")


def make_combined_normad_udr_nw() -> None:
    """Two-panel figure: NormAd NW US-default rate for LLaMA 8B and Gemma 4 12B."""

    model_specs = [
        ("LLaMA 3.1 8B",  "8b",     COLOR_WESTERN),
        ("Gemma 4 12B", "gemma4", COLOR_GEMMA4),
    ]

    panel_data = []
    for label, ms, color in model_specs:
        cult   = [load_normad_preds(c, ms) for c in CULT_CONDITIONS]
        nocult = [load_normad_preds(c, ms) for c in NOCULT_CONDITIONS]
        cult_vals   = [us_default_errors_nw(p, yn_only=True) for p in cult]
        nocult_vals = [us_default_errors_nw(p, yn_only=True) for p in nocult]
        panel_data.append((label, color, cult_vals, nocult_vals))

    # Shared y limits
    all_finite = [v for _, _, cv, nv in panel_data for v in cv + nv if not math.isnan(v)]
    mid  = (min(all_finite) + max(all_finite)) / 2
    half = 0.30 / 2

    x = np.arange(len(X_LABELS))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, (label, color, cult_vals, nocult_vals) in zip(axes, panel_data):
        muted = muted_color(color)
        ax.bar(x - width / 2, cult_vals,   width, color=color, label="Cultural Data",     zorder=3)
        ax.bar(x + width / 2, nocult_vals, width, color=muted, label="Non-Cultural Data", zorder=3)
        ax.set_title(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=11)
        ax.set_xlabel("Alignment Condition", fontsize=11)
        ax.set_ylabel("US-Default Rate Among Errors (US-Distant)", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(max(0, mid - half), mid + half)
        ax.set_xlim(-0.5, len(X_LABELS) - 0.5)

    from matplotlib.patches import Patch
    axes[1].legend(
        handles=[Patch(facecolor="dimgray",  label="Cultural Data"),
                 Patch(facecolor="lightgray", label="Non-Cultural Data")],
        loc="best", fontsize=10, frameon=True,
    )

    fig.tight_layout()
    out = FIGURES_DIR / "aya_cult_combined_normad_us_default_rate_nw.pdf"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def make_combined_normad_accuracy() -> None:
    """Four-panel figure (2×2): top row = BLEnD W/NW, bottom row = NormAd W/NW.
    Columns: LLaMA 3.1 8B | Gemma 4 12B. Each row shares a y-axis."""
    from matplotlib.lines import Line2D

    model_specs = [
        ("LLaMA 3.1 8B", "8b"),
        ("Gemma 4 12B",  "gemma4"),
    ]

    blend_data, normad_data = [], []
    for label, ms in model_specs:
        cult_na   = [load_normad_preds(c, ms) for c in CULT_CONDITIONS]
        nocult_na = [load_normad_preds(c, ms) for c in NOCULT_CONDITIONS]
        na_cult_acc   = [accuracy_by_group(p, yn_only=True)  for p in cult_na]
        na_nocult_acc = [accuracy_by_group(p, yn_only=True)  for p in nocult_na]
        normad_data.append((
            label,
            [d["Western"]     for d in na_cult_acc],
            [d["Non-Western"] for d in na_cult_acc],
            [d["Western"]     for d in na_nocult_acc],
            [d["Non-Western"] for d in na_nocult_acc],
            [_group_ci(p, "Western",     yn_only=True)  for p in cult_na],
            [_group_ci(p, "Non-Western", yn_only=True)  for p in cult_na],
            [_group_ci(p, "Western",     yn_only=True)  for p in nocult_na],
            [_group_ci(p, "Non-Western", yn_only=True)  for p in nocult_na],
        ))

        cult_bl   = [load_blend_preds(c, ms) for c in CULT_CONDITIONS]
        nocult_bl = [load_blend_preds(c, ms) for c in NOCULT_CONDITIONS]
        bl_cult_acc   = [accuracy_by_group(p, yn_only=False) for p in cult_bl]
        bl_nocult_acc = [accuracy_by_group(p, yn_only=False) for p in nocult_bl]
        blend_data.append((
            label,
            [d["Western"]     for d in bl_cult_acc],
            [d["Non-Western"] for d in bl_cult_acc],
            [d["Western"]     for d in bl_nocult_acc],
            [d["Non-Western"] for d in bl_nocult_acc],
            [_group_ci(p, "Western",     yn_only=False) for p in cult_bl],
            [_group_ci(p, "Non-Western", yn_only=False) for p in cult_bl],
            [_group_ci(p, "Western",     yn_only=False) for p in nocult_bl],
            [_group_ci(p, "Non-Western", yn_only=False) for p in nocult_bl],
        ))

    def row_ylim(rows):
        finite = [v for row in rows
                  for v in row[1] + row[2] + row[3] + row[4] if not math.isnan(v)]
        mid = (min(finite) + max(finite)) / 2
        return max(0, mid - 0.15), mid + 0.15

    blend_ylim  = row_ylim(blend_data)
    normad_ylim = row_ylim(normad_data)

    x = np.arange(len(X_LABELS))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey="row")

    row_specs = [
        (blend_data,  "BLEnD Accuracy",  blend_ylim),
        (normad_data, "NormAd Accuracy", normad_ylim),
    ]

    for row_idx, (panel_data, ylabel, ylim) in enumerate(row_specs):
        for col_idx, row in enumerate(panel_data):
            label, cw, cnw, nw, nnw, cw_ci, cnw_ci, nw_ci, nnw_ci = row
            ax = axes[row_idx][col_idx]
            for vals, ci, color, ls, marker, ms, lw in [
                (cw,  cw_ci,  COLOR_W_GROUP,  "-",  "o", 9, 2.6),
                (nw,  nw_ci,  COLOR_W_GROUP,  "--", "o", 7, 2.0),
                (cnw, cnw_ci, COLOR_NW_GROUP, "-",  "s", 9, 2.6),
                (nnw, nnw_ci, COLOR_NW_GROUP, "--", "s", 7, 2.0),
            ]:
                ax.fill_between(x, [lo for lo, _ in ci], [hi for _, hi in ci],
                                color=color, alpha=0.20)
                ax.plot(x, vals, color=color, linestyle=ls,
                        marker=marker, markersize=ms, linewidth=lw)
            if row_idx == 0:
                ax.set_title(label, fontsize=12)
            if row_idx == 1:
                ax.set_xlabel("Alignment Condition", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(X_LABELS, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.set_axisbelow(True)
            ax.set_ylim(*ylim)
            ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    color_handles = [
        Line2D([0], [0], color=COLOR_W_GROUP,  marker="o", markersize=8, linewidth=2, label="US-Centric"),
        Line2D([0], [0], color=COLOR_NW_GROUP, marker="s", markersize=8, linewidth=2, label="US-Distant"),
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="Cultural Data"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Non-Cultural Data"),
    ]
    leg1 = axes[1][1].legend(handles=color_handles, loc="upper right", fontsize=10,
                              frameon=True, title="Group")
    axes[1][1].add_artist(leg1)
    axes[1][1].legend(handles=style_handles, loc="lower left", fontsize=10,
                      frameon=True, title="Training Data")

    for row_idx, bench_label in enumerate(["BLEnD", "NormAd"]):
        axes[row_idx][1].yaxis.set_label_position("right")
        axes[row_idx][1].set_ylabel(bench_label, fontsize=13, fontweight="bold",
                                    rotation=270, labelpad=18)

    fig.tight_layout()
    out = FIGURES_DIR / "aya_cult_combined_normad_accuracy.pdf"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def make_figure_nlu_combined_models() -> None:
    """Single-panel NLU figure: BoolQ and CSQA for both LLaMA 3.1 8B and Gemma 4 12B.

    Color:   blue = LLaMA 3.1 8B, green = Gemma 4 12B
    Style:   solid = cultural data, dashed = non-cultural data
    Marker:  circle = BoolQ, square = CSQA
    """
    from matplotlib.lines import Line2D

    datasets    = [("boolq", "BoolQ", "o"), ("csqa", "CSQA", "^")]
    model_specs = [
        ("LLaMA 3.1 8B", "8b",     COLOR_WESTERN),
        ("Gemma 4 12B",  "gemma4", COLOR_GEMMA4),
    ]

    all_series = []
    all_finite = []
    for dataset, _, marker in datasets:
        for _, ms, color in model_specs:
            cult   = [_try_load_nlu(dataset, c, ms) for c in CULT_CONDITIONS]
            nocult = [_try_load_nlu(dataset, c, ms) for c in NOCULT_CONDITIONS]
            cv = [accuracy_overall(p) if p else float("nan") for p in cult]
            nv = [accuracy_overall(p) if p else float("nan") for p in nocult]
            cv_ci = [bootstrap_ci(p, accuracy_overall) if p else (float("nan"), float("nan")) for p in cult]
            nv_ci = [bootstrap_ci(p, accuracy_overall) if p else (float("nan"), float("nan")) for p in nocult]
            all_series.append((cv, nv, cv_ci, nv_ci, color, marker))
            all_finite += [v for v in cv + nv if not math.isnan(v)]

    shared_mid  = (min(all_finite) + max(all_finite)) / 2
    shared_half = max(max(all_finite) - min(all_finite), 0.02) * 1.4 / 2

    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cv, nv, cv_ci, nv_ci, color, marker in all_series:
        ax.fill_between(x, [lo for lo, _ in cv_ci], [hi for _, hi in cv_ci],
                        color=color, alpha=0.12)
        ax.fill_between(x, [lo for lo, _ in nv_ci], [hi for _, hi in nv_ci],
                        color=color, alpha=0.12)
        ax.plot(x, cv, marker=marker, markersize=11, linewidth=2.6,
                color=color, linestyle="-")
        ax.plot(x, nv, marker=marker, markersize=8, linewidth=2.0,
                color=color, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(shared_mid - shared_half, shared_mid + shared_half)
    ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    model_handles = [
        Line2D([0], [0], color=COLOR_WESTERN, linewidth=2.6, marker="o", markersize=8,
               label="LLaMA 3.1 8B"),
        Line2D([0], [0], color=COLOR_GEMMA4,  linewidth=2.6, marker="o", markersize=8,
               label="Gemma 4 12B"),
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2.6, label="Cultural Data"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="Non-Cultural Data"),
    ]
    bench_handles = [
        Line2D([0], [0], color="gray", linewidth=2.0, marker="o", markersize=10, label="BoolQ"),
        Line2D([0], [0], color="gray", linewidth=2.0, marker="^", markersize=10, label="CSQA"),
    ]

    leg1 = ax.legend(handles=model_handles,  loc="upper right", fontsize=10,
                     frameon=True, title="Model")
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=style_handles,  loc="lower left",  fontsize=10,
                     frameon=True, title="Training Data")
    ax.add_artist(leg2)
    ax.legend(handles=bench_handles, loc="lower right", fontsize=10,
              frameon=True, title="Benchmark")

    fig.tight_layout()
    out = FIGURES_DIR / "aya_cult_combined_nlu_accuracy.pdf"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def make_combined_benchmark_accuracy(model_size: str = "8b") -> None:
    """Single-panel overlay of NormAd and BLEnD W/NW accuracy.

    Color:      purple = Western, orange = Non-Western
    Line style: solid = cultural data, dashed = non-cultural data
    Marker:     circle = NormAd, square = BLEnD
    """
    from matplotlib.lines import Line2D

    ms  = model_size
    pfx = f"aya_cult_{ms}"

    cult_na   = [load_normad_preds(c, ms) for c in CULT_CONDITIONS]
    nocult_na = [load_normad_preds(c, ms) for c in NOCULT_CONDITIONS]
    cult_bl   = [load_blend_preds(c,  ms) for c in CULT_CONDITIONS]
    nocult_bl = [load_blend_preds(c,  ms) for c in NOCULT_CONDITIONS]

    na_cult_acc   = [accuracy_by_group(p, yn_only=True)  for p in cult_na]
    na_nocult_acc = [accuracy_by_group(p, yn_only=True)  for p in nocult_na]
    bl_cult_acc   = [accuracy_by_group(p, yn_only=False) for p in cult_bl]
    bl_nocult_acc = [accuracy_by_group(p, yn_only=False) for p in nocult_bl]

    x = np.arange(len(X_LABELS))
    fig, ax = plt.subplots(figsize=(7, 5.5))

    lines = [
        # (vals,                              color,         ls,   marker, ms, lw)
        ([d["Western"]     for d in na_cult_acc],   COLOR_W_GROUP,  "-",  "o", 9, 2.6),
        ([d["Western"]     for d in na_nocult_acc], COLOR_W_GROUP,  "--", "o", 7, 2.0),
        ([d["Non-Western"] for d in na_cult_acc],   COLOR_NW_GROUP, "-",  "o", 9, 2.6),
        ([d["Non-Western"] for d in na_nocult_acc], COLOR_NW_GROUP, "--", "o", 7, 2.0),
        ([d["Western"]     for d in bl_cult_acc],   COLOR_W_GROUP,  "-",  "s", 9, 2.6),
        ([d["Western"]     for d in bl_nocult_acc], COLOR_W_GROUP,  "--", "s", 7, 2.0),
        ([d["Non-Western"] for d in bl_cult_acc],   COLOR_NW_GROUP, "-",  "s", 9, 2.6),
        ([d["Non-Western"] for d in bl_nocult_acc], COLOR_NW_GROUP, "--", "s", 7, 2.0),
    ]
    for vals, color, ls, marker, msize, lw in lines:
        ax.plot(x, vals, color=color, linestyle=ls, marker=marker,
                markersize=msize, linewidth=lw)

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    all_vals = [v for vals, *_ in lines for v in vals if not math.isnan(v)]
    mid  = (min(all_vals) + max(all_vals)) / 2
    half = 0.30 / 2
    ax.set_ylim(mid - half, mid + half)
    ax.set_xlim(-0.4, len(X_LABELS) - 0.6)

    color_handles = [
        Line2D([0], [0], color=COLOR_W_GROUP,  linewidth=2, marker="o", markersize=8, label="US-Centric"),
        Line2D([0], [0], color=COLOR_NW_GROUP, linewidth=2, marker="o", markersize=8, label="US-Distant"),
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="Cultural Data"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Non-Cultural Data"),
    ]
    bench_handles = [
        Line2D([0], [0], color="gray", marker="o", markersize=8, linewidth=2, label="NormAd"),
        Line2D([0], [0], color="gray", marker="s", markersize=8, linewidth=2, label="BLEnD"),
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper right", fontsize=9,
                     frameon=True, title="Group")
    leg2 = ax.legend(handles=style_handles, loc="center right", fontsize=9,
                     frameon=True, title="Training Data")
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.legend(handles=bench_handles, loc="lower right", fontsize=9,
              frameon=True, title="Benchmark")

    fig.tight_layout()
    out = FIGURES_DIR / f"{pfx}_combined_accuracy.pdf"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
    make_figure_nlu_combined_models()
    make_combined_normad_udr_nw()
    make_combined_normad_accuracy()
    make_combined_benchmark_accuracy("8b")
