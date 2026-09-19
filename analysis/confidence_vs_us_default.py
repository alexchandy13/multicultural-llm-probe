"""Confidence vs US-default analysis.

For Non-Western predictions, bins examples by model confidence and asks:
does lower confidence predict higher P(pred == us_pred)?

Confidence = margin between top score and second-best score.
If the model uses US as a prior when uncertain, lower-confidence bins
should show higher US-match rates.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"

PIPELINE = [
    ("Base",           "base"),
    ("SFT cult",       "sft_aya_cult"),
    ("SFT+DPO cult",   "sftdpo_aya_cult"),
    ("SFT nocult",     "sft_aya_nocult"),
    ("SFT+DPO nocult", "sftdpo_aya_nocult"),
]

N_BINS = 4  # quartiles


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


def confidence_bins(preds: list[dict], score_key: str,
                    group: str = "Non-Western", yn_only: bool = False,
                    n_bins: int = N_BINS) -> list[dict]:
    """Bin NW predictions by confidence, return per-bin US-match rate."""
    subset = [p for p in preds
              if p.get("group") == group
              and p.get("country") != "US"
              and p.get("us_pred") is not None
              and p.get(score_key) is not None]
    if yn_only:
        subset = [p for p in subset if p["gold"] in ("yes", "no")]

    confs = [margin(p[score_key]) for p in subset]
    valid = [(c, p) for c, p in zip(confs, subset) if not math.isnan(c)]
    if not valid:
        return []

    valid.sort(key=lambda x: x[0])
    bin_size = len(valid) / n_bins
    bins = []
    for i in range(n_bins):
        lo = int(i * bin_size)
        hi = int((i + 1) * bin_size) if i < n_bins - 1 else len(valid)
        chunk = valid[lo:hi]
        conf_vals = [c for c, _ in chunk]
        us_match = sum(1 for _, p in chunk if p["pred"] == p["us_pred"])
        bins.append({
            "bin": i + 1,
            "conf_min": min(conf_vals),
            "conf_max": max(conf_vals),
            "conf_mean": statistics.mean(conf_vals),
            "n": len(chunk),
            "us_match_rate": us_match / len(chunk),
        })
    return bins


def fmt(v: float, w: int = 7) -> str:
    return f"{v:>{w}.3f}" if not math.isnan(v) else f"{'n/a':>{w}}"


def print_bins(bins: list[dict], label: str) -> None:
    if not bins:
        print(f"  {label}: no data")
        return
    n_bins = len(bins)
    bin_labels = ["low", "med-lo", "med-hi", "high"] if n_bins == 4 else [str(i+1) for i in range(n_bins)]
    print(f"\n  {label}")
    print(f"  {'Conf bin':<10} {'conf mean':>10} {'conf range':>22}  {'n':>6}  {'US-match rate':>14}")
    print(f"  {'─'*70}")
    for b, lbl in zip(bins, bin_labels):
        rng = f"[{b['conf_min']:.2f}, {b['conf_max']:.2f}]"
        print(f"  {lbl:<10} {fmt(b['conf_mean']):>10} {rng:>22}  {b['n']:>6}  {b['us_match_rate']:>13.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="8b", choices=["8b", "gemma4", "qwen35"])
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()
    ms = args.model_size

    print(f"\nConfidence vs US-match rate for Non-Western predictions  [{ms}]")
    print("Lower confidence → higher US-match rate would support the US-as-attractor hypothesis")

    for label, cond in PIPELINE:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

        preds_na = load_normad(cond, ms)
        if preds_na is not None:
            bins = confidence_bins(preds_na, "scores_batch_calibrated",
                                   yn_only=True, n_bins=args.bins)
            print_bins(bins, "NormAd (calibrated scores, yes/no only)")
        else:
            print("  NormAd: file missing")

        preds_bl = load_blend(cond, ms)
        if preds_bl is not None:
            # check if scores exist
            if any(p.get("scores") is not None for p in preds_bl):
                bins = confidence_bins(preds_bl, "scores", n_bins=args.bins)
                print_bins(bins, "BLEnD (raw scores)")
            else:
                print("  BLEnD: scores field missing in this file (older run)")
        else:
            print("  BLEnD: file missing")


if __name__ == "__main__":
    main()
