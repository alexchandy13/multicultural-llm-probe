"""Figures: US-match rate by confidence quartile for Non-Western predictions.

For each benchmark, bins NW predictions by confidence (margin between top
and 2nd-best log-prob score) and plots P(pred == us_pred) per bin.

If the model uses US as a prior when uncertain, the low-confidence quartile
should show the highest US-match rate.

Outputs:
  outputs/figures/confidence_us_default_normad_{model_size}.pdf
  outputs/figures/confidence_us_default_blend_{model_size}.pdf   (base only)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES_DIR    = PROJECT_ROOT / "outputs" / "figures"

# Match pipeline figure palette
COLOR_CULT   = "#4477AA"   # blue  — cult conditions
COLOR_NOCULT = "#CC6677"   # red   — nocult conditions
COLOR_BASE   = "#888888"   # gray  — base

PIPELINE = [
    ("Base",           "base",             COLOR_BASE,   "-",  "o", 7),
    ("SFT cult",       "sft_aya_cult",     COLOR_CULT,   "-",  "s", 7),
    ("SFT+DPO cult",   "sftdpo_aya_cult",  COLOR_CULT,   "--", "s", 7),
    ("SFT nocult",     "sft_aya_nocult",   COLOR_NOCULT, "-",  "^", 7),
    ("SFT+DPO nocult", "sftdpo_aya_nocult",COLOR_NOCULT, "--", "^", 7),
]

N_BINS = 4
BIN_LABELS = ["Low\nconfidence", "Med-low", "Med-high", "High\nconfidence"]


def margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return float("nan")
    s = sorted(scores, reverse=True)
    return s[0] - s[1]


def load_normad(cond: str, ms: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"normad_{cond}_{ms}_nfs_mpw_usprobe_batch_calibrated.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def load_blend(cond: str, ms: str) -> list[dict] | None:
    path = BEHAVIORAL_DIR / f"blend_{cond}_{ms}_nfs_usprobe.json"
    return json.loads(path.read_text())["predictions"] if path.exists() else None


def compute_bins(preds: list[dict], score_key: str,
                 yn_only: bool = False, n_bins: int = N_BINS,
                 correct: bool | None = None) -> list[float] | None:
    """Return US-match rate per confidence quartile for NW non-US predictions.

    correct=True  → correct predictions only
    correct=False → errors only
    correct=None  → all predictions (default)
    """
    subset = [p for p in preds
              if p.get("group") == "Non-Western"
              and p.get("country") != "US"
              and p.get("us_pred") is not None
              and p.get(score_key) is not None]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]
    if correct is True:
        subset = [p for p in subset if p["pred"] == p["gold"]]
    elif correct is False:
        subset = [p for p in subset if p["pred"] != p["gold"]]

    valid = [(margin(p[score_key]), p) for p in subset]
    valid = [(c, p) for c, p in valid if not math.isnan(c)]
    if not valid:
        return None

    valid.sort(key=lambda x: x[0])
    bin_size = len(valid) / n_bins
    rates = []
    for i in range(n_bins):
        lo = int(i * bin_size)
        hi = int((i + 1) * bin_size) if i < n_bins - 1 else len(valid)
        chunk = valid[lo:hi]
        us_match = sum(1 for _, p in chunk if p["pred"] == p["us_pred"])
        rates.append(us_match / len(chunk))
    return rates


def save_figure(lines: list[tuple], ylabel: str, out_path: Path,
                title: str = "") -> None:
    """lines: list of (label, rates, color, linestyle, marker, markersize)"""
    x = np.arange(N_BINS)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for label, rates, color, ls, marker, ms in lines:
        if rates is None:
            continue
        ax.plot(x, rates, color=color, linestyle=ls, marker=marker,
                markersize=ms, linewidth=2.2, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=10)
    ax.set_xlabel("Model Confidence (log-prob margin quartile)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.4, N_BINS - 0.6)

    all_vals = [v for _, rates, *_ in lines if rates for v in rates]
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 0.25, 0.03)
        ax.set_ylim(lo - pad, hi + pad)

    ax.legend(loc="best", fontsize=9, frameon=True)
    if title:
        ax.set_title(title, fontsize=11)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b",
                        choices=["8b", "gemma4", "qwen35"])
    args = parser.parse_args()
    ms = args.model_size

    # Each condition gets two lines: errors (solid) and correct (dashed)
    # so we can see if the confidence→US-match pattern differs by outcome.
    def build_lines(loader, score_key, yn_only=False):
        lines = []
        for label, cond, color, _, marker, msize in PIPELINE:
            preds = loader(cond, ms)
            if preds is None:
                continue
            if not any(p.get(score_key) is not None for p in preds):
                print(f"{label}: {score_key} missing (older run), skipping")
                continue
            err_rates = compute_bins(preds, score_key, yn_only=yn_only, correct=False)
            cor_rates = compute_bins(preds, score_key, yn_only=yn_only, correct=True)
            if err_rates:
                lines.append((f"{label} (errors)",  err_rates, color, "-",  marker, msize))
            if cor_rates:
                lines.append((f"{label} (correct)", cor_rates, color, ":", marker, msize - 2))
        return lines

    # ── NormAd ──────────────────────────────────────────────────────────────
    lines_na = build_lines(load_normad, "scores_batch_calibrated", yn_only=True)
    save_figure(
        lines_na,
        ylabel="P(pred = US pred)  [Non-Western]",
        out_path=FIGURES_DIR / f"confidence_us_default_normad_{ms}.pdf",
        title=f"NormAd — US-match rate by confidence quartile  [{ms}]",
    )

    # ── BLEnD (base only — SFT/DPO files predate scores field) ─────────────
    lines_bl = build_lines(load_blend, "scores")
    if lines_bl:
        save_figure(
            lines_bl,
            ylabel="P(pred = US pred)  [Non-Western]",
            out_path=FIGURES_DIR / f"confidence_us_default_blend_{ms}.pdf",
            title=f"BLEnD — US-match rate by confidence quartile  [{ms}]",
        )


if __name__ == "__main__":
    main()
