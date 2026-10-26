"""2x2 pipeline grid: benchmark x model, each panel US-centric vs US-distant
for both the cultural and the generic training arm.

Rows are the benchmark (NormAd, BLEnD), columns the model (Llama-3.1-8B,
Gemma-4-12B). Four lines per panel:

  colour      Western vs Non-Western, by the `group` field the paper used
  line style  dashed = cultural training data, solid = generic

Both arms share the Base point, so each panel is two forks from one origin.
Bands are 95% binomial intervals. All four panels share one identical y axis,
spanning the whole grid's data, so a slope means the same thing in every panel
and the vertical offsets between panels are real. That costs some detail --
BLEnD US-distant sits near 58 and NormAd US-centric near 94, about 36 points
apart -- so --per-panel-y restores the old behaviour of one window per panel.

    python analysis/pipeline_2x2_grid.py
    python analysis/pipeline_2x2_grid.py --format pdf --y-span 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import Counter

from analysis.cluster_accuracy_pipeline import find
from culnig.clusters import US_SIMILAR_CLUSTERS, cluster_of
from analysis.udr_pipeline import is_us
from analysis.nocult_pipeline_ci import (BENCH_LABEL, FIGURES, GROUPS,
                                         GROUP_COLOR, GROUP_LABEL, MODEL_LABEL,
                                         XLABELS, band, ci, find_for, group_of,
                                         set_span, two_group_acc)

BENCHES = ["normad", "blend"]

METRIC_LABEL = {
    "accuracy": "Accuracy (%)",
    "udr": "US-Default Rate Among Errors (%)",
    "kappa": "US-Default Agreement Above Chance (kappa)",
}
MODELS = ["8b", "gemma4"]

ARMS = {
    "cult":   (["base", "sft_aya_cult", "sftdpo_aya_cult"], "--", "Cultural"),
    "nocult": (["base", "sft_aya_nocult", "sftdpo_aya_nocult"], "-", "Generic"),
}


def two_group_udr(path, metric: str, partition: str = "group"):
    """{group: (value, n_errors)} for US-centric vs US-distant questions.

    US-anchored questions are dropped -- they match their own counterfactual by
    construction. "udr" is P(pred == us_pred | pred != gold); "kappa" corrects
    that for the agreement the error-conditioned label marginals force on their
    own, which matters because a checkpoint that collapses onto one label
    agrees with its counterfactual for free.
    """
    if path is None or not path.exists():
        return None
    pr = [x for x in json.loads(path.read_text())["predictions"]
          if x.get("us_pred") is not None and not is_us(x.get("country", ""))]
    out = {}
    for g in GROUPS:
        sub = [x for x in pr if group_of(x, partition) == g]
        err = [x for x in sub if x["pred"] != x["gold"]]
        if not err:
            continue
        agree = sum(x["pred"] == x["us_pred"] for x in err) / len(err)
        if metric == "udr":
            out[g] = (100 * agree, len(err))
        else:
            pm, um = Counter(x["pred"] for x in err), Counter(x["us_pred"]
                                                              for x in err)
            chance = sum(pm[c] * um[c] for c in set(pm) | set(um)) / len(err)**2
            k = (agree - chance) / (1 - chance) if chance < 1 else float("nan")
            out[g] = (100 * k, len(err))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png", choices=["png", "pdf", "jpg"])
    ap.add_argument("--metric", default="accuracy",
                    choices=["accuracy", "udr", "kappa"])
    ap.add_argument("--partition", default="group", choices=["group", "iw"],
                    help="group = the paper's Western/Non-Western field.")
    ap.add_argument("--groups", nargs="*", default=GROUPS, choices=GROUPS,
                    help="Which cluster groups to plot. Restricting to "
                         "us_distant drops the NormAd lines that rest on ~20 "
                         "errors and lets the shared axis tighten around what "
                         "is left.")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--y-span", type=float, default=25.0,
                    help="Minimum y-axis length in percentage points.")
    ap.add_argument("--per-panel-y", action="store_true",
                    help="Centre a separate y window on each panel's own data "
                         "instead of sharing one axis across the grid.")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    xs = range(len(XLABELS))
    extents, g_lo, g_hi = {}, [], []
    for row, bench in enumerate(BENCHES):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            lo, hi = [], []
            for arm, (conds, ls, _lbl) in ARMS.items():
                stages = [(two_group_acc(
                               find_for(bench, c, model, args.partition),
                               args.partition)
                           if args.metric == "accuracy"
                           else two_group_udr(
                               find_for(bench, c, model, args.partition),
                               args.metric, args.partition))
                          for c in conds]
                if any(s is None for s in stages):
                    continue
                for g in args.groups:
                    # For udr/kappa the count is ERRORS, not examples, so the
                    # same --min-n is a much harsher filter: NormAd's US-centric
                    # group has ~11-37 errors once US-anchored questions are
                    # dropped. Say so rather than letting a line vanish.
                    if not all(g in s and s[g][1] >= args.min_n
                               for s in stages):
                        have = [s[g][1] for s in stages if g in s]
                        print(f"    dropped {GROUP_LABEL[g]} / {_lbl} on "
                              f"{BENCH_LABEL[bench]} {MODEL_LABEL[model]}: "
                              f"n={have} below --min-n {args.min_n}")
                        continue
                    ys = [s[g][0] for s in stages]
                    half = [ci(s[g][0], s[g][1]) for s in stages]
                    band(ax, xs, ys, half, GROUP_COLOR[g], ls=ls)
                    # accuracy and udr are rates: a +-26 point band on a 20-error
                    # cell otherwise pushes the shared axis past 100%, which is
                    # both impossible and unreadable. kappa is not bounded this
                    # way, so it is left alone.
                    clip = args.metric in ("accuracy", "udr")
                    for y, h in zip(ys, half):
                        lo.append(max(y - h, 0.0) if clip else y - h)
                        hi.append(min(y + h, 100.0) if clip else y + h)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(XLABELS)
            ax.set_xlim(-0.25, len(XLABELS) - 0.75)
            ax.set_ylabel(METRIC_LABEL[args.metric])
            if row == len(BENCHES) - 1:
                ax.set_xlabel("Training Stage")
            ax.set_title(f"{BENCH_LABEL[bench]} — {MODEL_LABEL[model]}",
                         fontsize=11)
            ax.grid(alpha=0.25, linewidth=0.5, axis="y")
            if lo:
                extents[(row, col)] = (min(lo), max(hi))
                g_lo.append(min(lo))
                g_hi.append(max(hi))

    if args.per_panel_y:
        for (row, col), (lo, hi) in extents.items():
            set_span(axes[row][col], lo, hi, args.y_span)
    elif g_lo:
        # one axis for the whole grid, at least --y-span tall
        lo, hi = min(g_lo), max(g_hi)
        pad = max(0.0, (args.y_span - (hi - lo)) / 2) + 0.03 * (hi - lo)
        for ax in axes.ravel():
            ax.set_ylim(lo - pad, hi + pad)
        print(f"  shared y axis: {lo - pad:.1f} to {hi + pad:.1f} "
              f"({hi - lo + 2 * pad:.1f} pts)")

    # two keys rather than four combined entries: colour for the cluster group,
    # line style for which training data the arm used
    handles = [Line2D([], [], color=GROUP_COLOR[g], lw=1.8, label=GROUP_LABEL[g])
               for g in args.groups]
    handles += [Line2D([], [], color="0.35", lw=1.8, ls=ls, label=lbl)
                for _, ls, lbl in ARMS.values()]
    axes[0][0].legend(handles=handles, fontsize=8, ncol=2)

    who = " vs ".join(GROUP_LABEL[g] for g in args.groups)
    fig.suptitle(f"{who} {METRIC_LABEL[args.metric].split(' (')[0]} "
                 f"Through Post-Training (bands = 95% CI)", fontsize=12)
    fig.tight_layout()
    tag = "" if args.metric == "accuracy" else f"_{args.metric}"
    if list(args.groups) != GROUPS:
        tag += "_" + "_".join(args.groups)
    out = FIGURES / f"pipeline_2x2_centric_distant{tag}.{args.format}"
    fig.savefig(out, bbox_inches="tight",
                dpi=200 if args.format != "pdf" else None, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
