"""Bar graphs of NormAd accuracy DELTAS between condition pairs, split by US-similarity.

Emits a SINGLE multi-panel PDF: one panel per condition-pair comparison, two bars each:
  - LEFT (blue):  Δaccuracy for US-similar cluster group (EnglishSpeaking + ProtestantEurope)
  - RIGHT (red):  Δaccuracy for US-distant cluster group (all other I-W clusters)

Each panel auto-scales its own Y axis (no shared scale across panels), so the
lower bound can dip below zero when deltas are negative and the bar magnitudes
stay readable regardless of how small or large the per-comparison effect is.

Δaccuracy = accuracy(after) - accuracy(before). Positive bar = accuracy improved
after the alignment step; negative = accuracy dropped. The visual contrast between
the two bars shows whether alignment shifted the US-similar / US-distant balance.

Default comparisons (matches the paper's 4-condition framing):
  1. Base → SFT                     (effect of SFT alone)
  2. Base → DPO                     (effect of DPO alone)
  3. SFT  → SFT+DPO                 (effect of adding DPO on top of SFT)

Override defaults via --comparisons. Example:
  python3 analysis/accuracy_deltas_bars.py --comparisons base:sft base:dpo

Flags:
  --separate-files   also emit one PDF per comparison (in addition to the combined PDF)
  --shared-y         force a shared Y range across all panels (off by default)

Outputs:
  outputs/figures/accuracy_deltas_combined.pdf                (always)
  outputs/figures/accuracy_delta_{before}_to_{after}.pdf      (one per comparison, with --separate-files)
"""
from __future__ import annotations

import argparse
import csv
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
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

US_SIMILAR_CLUSTERS = {"EnglishSpeaking", "ProtestantEurope"}

COND_LABELS = {
    "base":   "Base",
    "sft":    "SFT",
    "dpo":    "DPO",
    "sftdpo": "SFT+DPO",
}

DEFAULT_COMPARISONS = [
    ("base", "sft"),
    ("base", "dpo"),
    ("sft", "sftdpo"),
]

COLOR_US_SIMILAR = "#4477AA"
COLOR_US_DISTANT = "#CC6677"


def load_country_to_cluster(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run analysis/culturemapping/compute_iw_coords.py first.")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["normad_country"]:
                out[r["normad_country"]] = r["cluster"]
    return out


def group_accuracy(cond: str, country_to_cluster: dict, size_suffix: str = "") -> dict[str, float]:
    """Return {'US-similar': acc, 'US-distant': acc} for one condition.

    Accuracy is pooled across all predictions in the group (not country-mean).
    """
    path = BEHAVIORAL_DIR / f"normad_{cond}{size_suffix}.json"
    if not path.exists():
        return {}
    preds = json.loads(path.read_text()).get("predictions", [])
    by_group = defaultdict(lambda: [0, 0])  # group → [correct, total]
    for p in preds:
        country = p.get("country", "")
        cluster = country_to_cluster.get(country) or country_to_cluster.get(country.lower())
        if cluster is None:
            continue
        group = "US-Centric/Western" if cluster in US_SIMILAR_CLUSTERS else "US-Distant/Non-Western"
        by_group[group][0] += int(p["gold"] == p["pred"])
        by_group[group][1] += 1
    return {g: (c / t) for g, (c, t) in by_group.items() if t > 0}


def parse_comparisons(specs: list[str]) -> list[tuple[str, str]]:
    """Parse "before:after" strings into (before, after) tuples."""
    out = []
    for s in specs:
        if ":" not in s:
            sys.exit(f"--comparisons entry {s!r} must be of form before:after")
        before, after = s.split(":", 1)
        out.append((before.strip(), after.strip()))
    return out


def label_for_comparison(before: str, after: str) -> str:
    return f"{COND_LABELS.get(before, before)}\n→ {COND_LABELS.get(after, after)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=["3b", "8b", "gemma4"], default="3b")
    parser.add_argument(
        "--comparisons", nargs="+", default=None,
        help="List of 'before:after' condition pairs (e.g. base:sft). "
             "Defaults to the paper's 3 canonical comparisons.",
    )
    parser.add_argument("--no-values", action="store_true",
                        help="Hide numeric value annotations on top of bars.")
    parser.add_argument("--separate-files", action="store_true",
                        help="Also emit one PDF per comparison in addition to the combined multi-panel PDF.")
    parser.add_argument("--shared-y", action="store_true",
                        help="Force a shared Y-axis range across all panels "
                             "(off by default — each panel auto-scales independently).")
    args = parser.parse_args()

    size_suffix = "" if args.model_size == "3b" else f"_{args.model_size}"
    fig_size_suffix = f"_{args.model_size}"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = parse_comparisons(args.comparisons) if args.comparisons else DEFAULT_COMPARISONS

    country_to_cluster = load_country_to_cluster(IW_COORDS)

    # Compute group accuracies for every condition that appears in any comparison.
    needed_conds = {c for pair in comparisons for c in pair}
    cond_accs: dict[str, dict[str, float]] = {}
    for cond in needed_conds:
        cond_accs[cond] = group_accuracy(cond, country_to_cluster, size_suffix)
        if not cond_accs[cond]:
            print(f"[warn] no data for condition {cond!r}; will appear as NaN", file=sys.stderr)

    # Compute deltas for each comparison
    sim_deltas, dist_deltas, labels = [], [], []
    for before, after in comparisons:
        b = cond_accs.get(before, {})
        a = cond_accs.get(after, {})
        sim_deltas.append(a.get("US-Centric/Western", float("nan")) - b.get("US-Centric/Western", float("nan")))
        dist_deltas.append(a.get("US-Distant/Non-Western", float("nan")) - b.get("US-Distant/Non-Western", float("nan")))
        labels.append(label_for_comparison(before, after))

    # Determine shared y-range if requested
    all_vals = [v for v in sim_deltas + dist_deltas if not math.isnan(v)]
    global_amax = max(abs(v) for v in all_vals) if all_vals else 0.05
    y_lim = (-global_amax * 1.25, global_amax * 1.25) if args.shared_y else None

    def draw_panel(ax, before, after, sd, dd, *, title_fs=10, xlabel_fs=9,
                   value_fs=9, force_y_lim=None):
        """Render one delta-comparison panel onto ax."""
        bars = ax.bar([0, 1], [sd, dd], 0.6,
                      color=[COLOR_US_SIMILAR, COLOR_US_DISTANT],
                      edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.6, zorder=2)

        # Per-panel auto-scaling: lower bound dips below zero if any delta is
        # negative; upper bound clears the largest positive delta. We pad by
        # ~15% of the panel's local magnitude so value annotations fit.
        finite = [v for v in (sd, dd) if not math.isnan(v)]
        local_amax = max(abs(v) for v in finite) if finite else 0.05
        if force_y_lim is not None:
            ax.set_ylim(*force_y_lim)
            pad = (force_y_lim[1] - force_y_lim[0]) * 0.02
        else:
            lo = min(finite + [0.0]) if finite else -0.05
            hi = max(finite + [0.0]) if finite else 0.05
            span = max(hi - lo, local_amax * 0.5, 0.02)
            ax.set_ylim(lo - span * 0.18, hi + span * 0.18)
            pad = local_amax * 0.04

        if not args.no_values:
            for bar, v in zip(bars, [sd, dd]):
                if math.isnan(v): continue
                offset = pad if v >= 0 else -pad
                va = "bottom" if v >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                        f"{v:+.3f}", ha="center", va=va, fontsize=value_fs)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Western", "Non-Western"], fontsize=xlabel_fs)
        ax.set_title(label_for_comparison(before, after), fontsize=title_fs)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    # === COMBINED MULTI-PANEL (always emitted) ===
    n = len(comparisons)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.8 * rows),
                             sharey=False, squeeze=False)
    for i, ((before, after), sd, dd) in enumerate(zip(comparisons, sim_deltas, dist_deltas)):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        draw_panel(ax, before, after, sd, dd,
                   title_fs=10, xlabel_fs=9, value_fs=9,
                   force_y_lim=y_lim)
        ax.set_ylabel("Δ NormAd accuracy")
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / f"accuracy_deltas_combined{fig_size_suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # === OPTIONAL: ONE FIGURE PER COMPARISON ===
    if args.separate_files:
        for (before, after), sd, dd in zip(comparisons, sim_deltas, dist_deltas):
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            draw_panel(ax, before, after, sd, dd,
                       title_fs=11, xlabel_fs=10, value_fs=10,
                       force_y_lim=y_lim)
            ax.set_ylabel("Δ NormAd accuracy (after − before)")
            fig.tight_layout()
            out = FIGURES_DIR / f"accuracy_delta_{before}_to_{after}{fig_size_suffix}.pdf"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {out}")

    # Console summary
    print(f"\n  {'comparison':<35s}  {'US-sim Δ':>10s}  {'US-dist Δ':>10s}  {'gap-shift':>10s}")
    for (before, after), sd, dd in zip(comparisons, sim_deltas, dist_deltas):
        gap_shift = sd - dd
        label = f"{before} → {after}"
        sd_s = f"{sd:+.3f}" if not math.isnan(sd) else "nan"
        dd_s = f"{dd:+.3f}" if not math.isnan(dd) else "nan"
        gs_s = f"{gap_shift:+.3f}" if not math.isnan(gap_shift) else "nan"
        print(f"  {label:<35s}  {sd_s:>10s}  {dd_s:>10s}  {gs_s:>10s}")
    print(f"\n  gap-shift > 0  →  alignment widened the US-favoring gap")
    print(f"  gap-shift < 0  →  alignment narrowed it")


if __name__ == "__main__":
    main()
