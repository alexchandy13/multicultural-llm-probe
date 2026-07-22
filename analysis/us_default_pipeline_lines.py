"""Line graph of US-default rate among errors across an alignment pipeline.

US-default rate = wrong_us_match / (wrong_us_match + wrong_us_diverge),
computed only over examples the model got wrong. One line per model;
all models on the same chart.

Reads *_yn_usprobe.json (or *_fsN_yn_usprobe.json) files produced by
eval_normad.py --yn-only --us-probe.

Examples:
  python3 analysis/us_default_pipeline_lines.py --few-shot 2 --conditions base sft dpo sftdpo
  python3 analysis/us_default_pipeline_lines.py --conditions base sft dpo sftdpo

Outputs:
  outputs/figures/us_default_pipeline_lines{sfx}.pdf
"""
from __future__ import annotations

import argparse
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
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

COND_LABELS = {
    "base":   "Base",
    "sft":    "SFT",
    "dpo":    "DPO",
    "sftdpo": "SFT+DPO",
}

DEFAULT_PIPELINE = ["base", "sft", "sftdpo"]

MODEL_COLORS = {
    "8b":     "#4477AA",
    "gemma4": "#CC6677",
    "3b":     "#117733",
    "qwen35": "#DDCC77",
}
MODEL_LABELS = {
    "8b":     "Llama 8B",
    "gemma4": "Gemma4",
    "3b":     "Llama 3B",
    "qwen35": "Qwen3.5",
}
MARKERS = ["o", "s", "^", "D"]


def overall_udr(cond: str, size_suffix: str) -> float | None:
    """Return overall US-default rate for one condition+model, or None if missing."""
    path = BEHAVIORAL_DIR / f"normad_{cond}{size_suffix}.json"
    if not path.exists():
        return None
    preds = json.loads(path.read_text()).get("predictions", [])
    wum = wud = 0
    for p in preds:
        if p.get("us_pred") is None:
            continue
        if p["pred"] == p["gold"]:
            continue
        if p["pred"] == p["us_pred"]:
            wum += 1
        else:
            wud += 1
    total = wum + wud
    return wum / total if total > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--model-size",
                        default="8b",
                        choices=["3b", "8b", "gemma4", "qwen35"])
    parser.add_argument("--few-shot", type=int, default=0, metavar="N")
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--y-from-zero", action="store_true")
    args = parser.parse_args()

    fs_sfx = f"_fs{args.few_shot}" if args.few_shot > 0 else ""
    fig_sfx = fs_sfx + args.out_suffix
    conditions = args.conditions if args.conditions else DEFAULT_PIPELINE
    if len(conditions) < 2:
        sys.exit("Need at least 2 conditions.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    size = args.model_size
    size_suffix = ("" if size == "3b" else f"_{size}") + fs_sfx + "_yn_usprobe"
    udrs = []
    for cond in conditions:
        v = overall_udr(cond, size_suffix)
        if v is None:
            print(f"[warn] missing: normad_{cond}{size_suffix}.json", file=sys.stderr)
        udrs.append(v if v is not None else float("nan"))

    color = MODEL_COLORS.get(size, "#333333")
    label = MODEL_LABELS.get(size, size)

    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(conditions)), 5.5))

    ax.plot(x, udrs, marker="o", markersize=9, linewidth=2.4, color=color, label=label)
    for xi, v in zip(x, udrs):
        if not math.isnan(v):
            ax.annotate(f"{v:.1%}", (xi, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS.get(c, c) for c in conditions], fontsize=11)
    ax.set_xlabel("Alignment Condition", fontsize=11)
    ax.set_ylabel("US-Default Rate Among Errors", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    shot_label = f"fs{args.few_shot}" if args.few_shot > 0 else "0-shot"
    ax.set_title(f"US-Default Rate Among Errors — {label} ({shot_label})", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    finite = [v for v in udrs if not math.isnan(v)]
    if finite:
        if args.y_from_zero:
            ax.set_ylim(0, min(max(finite) * 1.10, 1.0))
        else:
            lo, hi = min(finite), max(finite)
            span = max(hi - lo, 0.02)
            ax.set_ylim(max(0, lo - span * 0.15), min(1.0, hi + span * 0.15))

    ax.set_xlim(-0.4, len(conditions) - 0.6)
    fig.tight_layout()

    out = FIGURES_DIR / f"us_default_pipeline_lines_{size}{fig_sfx}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    print(f"\n  {'condition':<18s}  {'udr':>8}")
    for j, cond in enumerate(conditions):
        v = udrs[j]
        row = f"{v:>7.1%}" if not math.isnan(v) else "    nan"
        print(f"  {cond:<18s}  {row}")


if __name__ == "__main__":
    main()
