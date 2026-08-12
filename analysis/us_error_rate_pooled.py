"""Line figure: pooled US-error rate across the alignment pipeline.

Pools cult + nocult predictions at each stage (Base / SFT / SFT+DPO) and plots
the US-default rate among errors for NormAd and BLEnD, both models on one figure.

Output:
  outputs/figures/us_error_rate_pooled.pdf

Usage:
    python analysis/us_error_rate_pooled.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES_DIR    = PROJECT_ROOT / "outputs" / "figures"

X_LABELS = ["Base", "SFT", "SFT+DPO"]

STAGES = {
    "Base":    ["base"],
    "SFT":     ["sft_aya_cult",    "sft_aya_nocult"],
    "SFT+DPO": ["sftdpo_aya_cult", "sftdpo_aya_nocult"],
}

COLOR_8B     = "#4477AA"   # blue
COLOR_GEMMA  = "#CC6677"   # red


def load(p: Path) -> list[dict]:
    return json.loads(p.read_text())["predictions"]


def us_err(preds: list[dict], yn_only: bool) -> float:
    errors = [p for p in preds
              if p.get("country") != "US"
              and p.get("us_pred") is not None
              and p["pred"] != p["gold"]]
    if yn_only:
        errors = [p for p in errors if p["gold"] in ("yes", "no")]
    return sum(1 for p in errors if p["pred"] == p["us_pred"]) / len(errors) if errors else float("nan")


def collect(benchmark: str, model_size: str, yn_only: bool) -> list[float]:
    vals = []
    for conds in STAGES.values():
        preds = []
        for c in conds:
            if benchmark == "normad":
                p = BEHAVIORAL_DIR / f"normad_{c}_{model_size}_nfs_mpw_usprobe_batch_calibrated.json"
            else:
                p = BEHAVIORAL_DIR / f"blend_{c}_{model_size}_nfs_usprobe.json"
            if p.exists():
                preds += load(p)
        vals.append(us_err(preds, yn_only) if preds else float("nan"))
    return vals


def main() -> None:
    x = np.arange(len(X_LABELS))

    normad_8b    = collect("normad", "8b",     yn_only=True)
    normad_gemma = collect("normad", "gemma4", yn_only=True)
    blend_8b     = collect("blend",  "8b",     yn_only=False)
    blend_gemma  = collect("blend",  "gemma4", yn_only=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (title, vals_8b, vals_gemma, yn) in zip(axes, [
        ("NormAd", normad_8b, normad_gemma, True),
        ("BLEnD",  blend_8b,  blend_gemma,  False),
    ]):
        ax.plot(x, vals_8b,    marker="o", markersize=8, linewidth=2.4,
                color=COLOR_8B,    linestyle="-",  label="LLaMA 3 8B")
        ax.plot(x, vals_gemma, marker="s", markersize=8, linewidth=2.4,
                color=COLOR_GEMMA, linestyle="-",  label="Gemma 4 12B")

        for xi, (v8, vg) in enumerate(zip(vals_8b, vals_gemma)):
            ax.annotate(f"{v8:.3f}",  (xi, v8),  textcoords="offset points",
                        xytext=(0, 10),  ha="center", fontsize=9, color=COLOR_8B)
            ax.annotate(f"{vg:.3f}", (xi, vg), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=9, color=COLOR_GEMMA)

        all_vals = [v for v in vals_8b + vals_gemma if v == v]
        if all_vals:
            mid  = (min(all_vals) + max(all_vals)) / 2
            half = max(max(all_vals) - min(all_vals), 0.02) * 1.4 / 2
            ax.set_ylim(mid - half, mid + half)

        ax.set_xticks(x)
        ax.set_xticklabels(X_LABELS, fontsize=11)
        ax.set_xlim(-0.4, len(X_LABELS) - 0.6)
        ax.set_xlabel("Alignment Stage", fontsize=11)
        ax.set_ylabel("US-Default Rate Among Errors", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, frameon=True)

    fig.suptitle("Pooled US-Default Error Rate Across Alignment Pipeline",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "us_error_rate_pooled.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
