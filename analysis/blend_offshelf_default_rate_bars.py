"""Default rate among errors, 8 probe countries x 4 off-the-shelf instruct
models, BLEnD only.

One panel per model, 8 bars per panel (one per probe country), colored by IW
cluster via the same CLUSTER_COLORS used elsewhere in this project. Default
rate among errors = of the model's wrong answers (on examples not already
from that probe country), the fraction whose prediction matches what the
model predicts when that country is substituted into the prompt.

Uses the current raw-prompt instruct probe files (_nfs ... probe_raw.json),
which load_probe_file in cluster_probe_analysis.py does not look for (it has
no _raw suffix), so files are loaded directly here instead.

    python analysis/blend_offshelf_default_rate_bars.py
"""
from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.cluster_accuracy_pipeline import CLUSTER_COLORS
from analysis.cluster_probe_analysis import (default_rate_among_errors, probe_slug,
                                             same_country)

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES = PROJECT_ROOT / "outputs" / "figures" / "blend_defaults"

# Same 8 countries as the project-wide cluster-representative set (CLUSTER_REPS
# + US), in cluster order. These are the ones with _raw probe files run for all
# four off-the-shelf instruct models -- BLEnD's own native cluster reps are a
# different, narrower 7-country set and don't have this grid.
ORDER = ["EnglishSpeaking", "ProtestantEurope", "CatholicEurope", "Orthodox",
         "LatinAmerica", "Confucian", "SouthAsia", "AfricanIslamic"]
REPS = {
    "EnglishSpeaking": "us", "ProtestantEurope": "netherlands",
    "CatholicEurope": "hungary", "Orthodox": "bosnia_and_herzegovina",
    "LatinAmerica": "mexico", "Confucian": "taiwan",
    "SouthAsia": "philippines", "AfricanIslamic": "iran",
}
COUNTRY_LABEL = {
    "us": "US", "netherlands": "Netherlands", "hungary": "Hungary",
    "bosnia_and_herzegovina": "Bosnia & Herz.", "mexico": "Mexico",
    "taiwan": "Taiwan", "philippines": "Philippines", "iran": "Iran",
}

MODELS = [
    ("8b_instruct",     "base",     "Llama-3.1-8B-Instruct"),
    ("gemma4_instruct", "base",     "Gemma-4-12B-it"),
    ("olmo3_instruct",  "offshelf", "Olmo-3-7B-Instruct"),
    ("qwen35_instruct", "offshelf", "Qwen3.5-9B"),
]


def load_raw_probe(condition: str, model: str, country: str) -> list[dict] | None:
    slug = probe_slug(country)
    path = BEHAVIORAL / f"blend_{condition}_{model}_nfs_{slug}probe_raw.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["predictions"]


def default_rate_all_rows(preds: list[dict], probe_country_name: str) -> tuple[float, int]:
    """P(pred == probe_pred) over ALL non-probe-country rows, not just errors."""
    rows = [p for p in preds if p.get("us_pred") is not None
            and not same_country(p["country"], probe_country_name)]
    if not rows:
        return float("nan"), 0
    return sum(1 for p in rows if p["pred"] == p["us_pred"]) / len(rows), len(rows)


def ci95(rate_pct: float, n: int) -> float:
    p = rate_pct / 100
    return 100 * 1.96 * sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png", choices=["png", "pdf", "jpg"])
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--errors-only", action="store_true",
                    help="Default rate among errors only, instead of all rows (default).")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4.4), sharey=True)
    xs = range(len(ORDER))

    for ax, (model, condition, label) in zip(axes, MODELS):
        rates, halfs, colors, labels = [], [], [], []
        for cl in ORDER:
            country = REPS[cl]
            preds = load_raw_probe(condition, model, country)
            if preds is None:
                print(f"  missing: blend_{condition}_{model}_nfs_{probe_slug(country)}probe_raw.json")
                rates.append(0); halfs.append(0)
                colors.append(CLUSTER_COLORS[cl]); labels.append(COUNTRY_LABEL[country])
                continue
            if args.errors_only:
                rate, _, n_match, n_other = default_rate_among_errors(preds, country)
                n = n_match + n_other
            else:
                rate, n = default_rate_all_rows(preds, country)
            if n < args.min_n:
                print(f"  {label} / {COUNTRY_LABEL[country]}: only {n} rows, below --min-n {args.min_n}")
            pct = 100 * rate if rate == rate else 0.0
            rates.append(pct)
            halfs.append(ci95(pct, n))
            colors.append(CLUSTER_COLORS[cl])
            labels.append(COUNTRY_LABEL[country])

        ax.bar(xs, rates, yerr=halfs, color=colors, width=0.65,
               capsize=3, error_kw={"linewidth": 1, "ecolor": "#555555"})
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5, axis="y")
        ax.set_ylim(0, 100)

    metric_label = "Default Rate Among Errors (%)" if args.errors_only else "Default Rate, All Rows (%)"
    axes[0].set_ylabel(metric_label)
    suffix = "errors" if args.errors_only else "allrows"
    fig.suptitle(f"BLEnD — {metric_label}, 8 Probe Countries, Off-the-Shelf Instruct Models",
                 fontsize=12)
    fig.tight_layout()
    out = FIGURES / f"blend_offshelf_default_rate_8country_{suffix}.{args.format}"
    fig.savefig(out, bbox_inches="tight",
                dpi=200 if args.format != "pdf" else None, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
