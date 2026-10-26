"""2x2 of cluster-default rate among errors: benchmark x model, generic arm.

Rows are the benchmark (NormAd, BLEnD), columns the model. Every panel shows
the same eight probe countries, one line each in a fixed colour, through
Base -> SFT -> SFT+DPO on the generic (non-cultural) training arm.

The default rate is: of the model's errors, the fraction whose prediction
matches what it predicts when that probe country is substituted into the
prompt. Probe-country questions are excluded from their own line, since they
match their own counterfactual by construction.

Bands are 95% binomial intervals on the ERROR count, which is far smaller than
the example count -- NormAd gives ~150-400 errors per cell against BLEnD's
~8,000 -- so NormAd's bands are several times wider and some of its apparent
movement is inside them.

Uses the shot variants whose probe grids are complete -- NormAd _fs2_yn and
BLEnD _nfs. The submitted paper's NormAd setting (_nfs_mpw) only ever had
probes run for Taiwan and Iran, so an eight-country version of this figure is
not available at that setting.

    python analysis/probe_default_2x2.py
    python analysis/probe_default_2x2.py --arm aya_cult
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from math import sqrt

from analysis.cluster_probe_accuracy_lines import (ARMS, ARM_LABEL,
                                                   BENCH_TITLE, FIGURES,
                                                   MODEL_LABEL, ORDER, REPS,
                                                   label_for)
from analysis.cluster_accuracy_pipeline import CLUSTER_COLORS
from analysis.cluster_probe_analysis import (default_rate_among_errors,
                                             load_probe_file)

BENCHES = ["normad", "blend"]
MODELS = ["8b", "gemma4"]


def cdr_with_n(bench: str, model: str, conds: list[str]):
    """{cluster: ([rate% per stage], [n_errors per stage])}."""
    out = {}
    for cl in ORDER:
        rates, ns = [], []
        for cond in conds:
            pr = load_probe_file(cond, model, REPS[cl], benchmark=bench)
            if pr is None:
                rates.append(None)
                ns.append(0)
                continue
            rate, _, n_match, n_other = default_rate_among_errors(pr, REPS[cl])
            rates.append(100 * rate)
            ns.append(n_match + n_other)
        if all(r is not None for r in rates):
            out[cl] = (rates, ns)
    return out


def ci(p_pct: float, n: int) -> float:
    p = p_pct / 100
    return 100 * 1.96 * sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="aya_nocult", choices=list(ARMS))
    ap.add_argument("--format", default="png", choices=["png", "pdf", "jpg"])
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--shared-y", action="store_true",
                    help="One y axis across the grid. Off by default: NormAd "
                         "runs 35-90% and BLEnD 60-72%, so a shared axis "
                         "flattens both.")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    xlabels, conds = ARMS[args.arm]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    xs = range(len(xlabels))
    seen, g_lo, g_hi = {}, [], []

    for row, bench in enumerate(BENCHES):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            data = cdr_with_n(bench, model, conds)
            for cl, (cdr, ns) in data.items():
                if min(ns) < args.min_n:
                    print(f"    {BENCH_TITLE[bench]} {MODEL_LABEL[model]}: "
                          f"{label_for(cl)} has only {min(ns)} errors, "
                          f"below --min-n {args.min_n}")
                    continue
                half = [ci(v, n) for v, n in zip(cdr, ns)]
                col = CLUSTER_COLORS[cl]
                ax.plot(xs, cdr, color=col, marker="o", markersize=5,
                        linewidth=1.8)
                ax.fill_between(xs, [max(v - h, 0) for v, h in zip(cdr, half)],
                                [min(v + h, 100) for v, h in zip(cdr, half)],
                                color=col, alpha=0.12, linewidth=0)
                seen[cl] = True
                g_lo += [max(v - h, 0) for v, h in zip(cdr, half)]
                g_hi += [min(v + h, 100) for v, h in zip(cdr, half)]
            ax.set_xticks(list(xs))
            ax.set_xticklabels(xlabels)
            ax.set_xlim(-0.25, len(xlabels) - 0.75)
            ax.set_ylabel("Default Rate Among Errors (%)")
            if row == len(BENCHES) - 1:
                ax.set_xlabel("Training Stage")
            ax.set_title(f"{BENCH_TITLE[bench]} — {MODEL_LABEL[model]}",
                         fontsize=11)
            ax.grid(alpha=0.25, linewidth=0.5, axis="y")

    if args.shared_y and g_lo:
        lo, hi = min(g_lo), max(g_hi)
        pad = 0.04 * (hi - lo)
        for ax in axes.ravel():
            ax.set_ylim(lo - pad, hi + pad)
        print(f"  shared y axis: {lo - pad:.1f} to {hi + pad:.1f}")

    handles = [Line2D([], [], color=CLUSTER_COLORS[cl], lw=1.8, marker="o",
                      markersize=5, label=label_for(cl)) for cl in seen]
    axes[0][0].legend(handles=handles, fontsize=7.5, ncol=2,
                      title="Probe Country", title_fontsize=8)

    fig.suptitle(f"Cluster-Default Rate Among Errors — {ARM_LABEL[args.arm]}",
                 fontsize=12)
    fig.tight_layout()
    out = FIGURES / f"probe_default_2x2_{args.arm}.{args.format}"
    fig.savefig(out, bbox_inches="tight",
                dpi=200 if args.format != "pdf" else None, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
