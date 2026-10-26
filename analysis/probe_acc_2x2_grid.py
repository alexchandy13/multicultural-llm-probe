"""2x2 grid of swapped-country accuracy, probe countries pooled by proximity.

Rows are the benchmark, columns the model. The metric is "swapped accuracy":
substitute one country into EVERY question and score against the original gold,
i.e. how well you would do by assuming the whole benchmark was about that
country. Four lines per panel:

  colour      US-centric probe countries (US, Netherlands) vs US-distant
              (Hungary, Bosnia, Mexico, Taiwan, Philippines, Iran)
  line style  dashed = cultural training data, solid = generic

Two deliberate departures from the submitted paper, both forced:

  * The paper's Western/Non-Western `group` field cannot partition the PROBE
    countries -- Hungary and Bosnia are "Other", and on BLEnD only the US,
    Mexico and Iran appear as question countries at all. So the probe split is
    IW cluster membership (US_SIMILAR_CLUSTERS) instead.
  * The paper's NormAd runs are _nfs_mpw, but probes under that setting exist
    only for Taiwan and Iran. The complete probe grid is _fs2_yn for NormAd and
    _nfs for BLEnD, so those are used here.

Bands are binomial on ONE country's worth of examples, not the pooled total:
the same question is re-scored once per probe country, so pooling 6 countries
would treat 6 correlated observations of the same item as independent and
report a band about sqrt(6) too narrow.

    python analysis/probe_acc_2x2_grid.py
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

from analysis.cluster_probe_analysis import (CLUSTER_REPS, PROBE_SUFFIX,
                                             load_probe_file, probe_accuracy,
                                             same_country)
from analysis.nocult_pipeline_ci import (BENCH_LABEL, FIGURES, GROUP_COLOR,
                                         MODEL_LABEL, XLABELS, band, ci,
                                         set_span)
from culnig.clusters import US_SIMILAR_CLUSTERS

BENCHES = ["normad", "blend"]
MODELS = ["8b", "gemma4"]

# probe country -> which side of the split, by its cluster's proximity to the US
REPS = {"EnglishSpeaking": "us", **CLUSTER_REPS}
PROBE_GROUP = {c: ("us_centric" if cl in US_SIMILAR_CLUSTERS else "us_distant")
               for cl, c in REPS.items()}
GROUPS = ["us_centric", "us_distant"]
GROUP_LABEL = {"us_centric": "US-Centric Probe Countries",
               "us_distant": "US-Distant Probe Countries"}

ARMS = {
    "cult":   (["base", "sft_aya_cult", "sftdpo_aya_cult"], "--", "Cultural"),
    "nocult": (["base", "sft_aya_nocult", "sftdpo_aya_nocult"], "-", "Generic"),
}


def grouped_probe_acc(bench: str, cond: str, model: str):
    """{group: (mean swapped accuracy %, per-country n)} or None."""
    out = {}
    for g in GROUPS:
        rates, ns = [], []
        for country, grp in PROBE_GROUP.items():
            if grp != g:
                continue
            pr = load_probe_file(cond, model, country, benchmark=bench)
            if pr is None:
                continue
            elig = [x for x in pr if x.get("us_pred") is not None
                    and not same_country(x["country"], country)]
            if not elig:
                continue
            rates.append(100 * probe_accuracy(pr, country))
            ns.append(len(elig))
        if rates:
            out[g] = (sum(rates) / len(rates), sum(ns) // len(ns))
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png", choices=["png", "pdf", "jpg"])
    ap.add_argument("--y-span", type=float, default=25.0)
    ap.add_argument("--per-panel-y", action="store_true")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    xs = range(len(XLABELS))
    extents, g_lo, g_hi = {}, [], []
    for row, bench in enumerate(BENCHES):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            lo, hi = [], []
            for arm, (conds, ls, _) in ARMS.items():
                stages = [grouped_probe_acc(bench, c, model) for c in conds]
                if any(s is None for s in stages):
                    continue
                for g in GROUPS:
                    if not all(g in s for s in stages):
                        continue
                    ys = [s[g][0] for s in stages]
                    half = [ci(s[g][0], s[g][1]) for s in stages]
                    band(ax, xs, ys, half, GROUP_COLOR[g], ls=ls)
                    lo += [max(y - h, 0.0) for y, h in zip(ys, half)]
                    hi += [min(y + h, 100.0) for y, h in zip(ys, half)]
            ax.set_xticks(list(xs))
            ax.set_xticklabels(XLABELS)
            ax.set_xlim(-0.25, len(XLABELS) - 0.75)
            ax.set_ylabel("Swapped-Country Accuracy (%)")
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
        for (r, c), (lo, hi) in extents.items():
            set_span(axes[r][c], lo, hi, args.y_span)
    elif g_lo:
        lo, hi = min(g_lo), max(g_hi)
        pad = max(0.0, (args.y_span - (hi - lo)) / 2) + 0.03 * (hi - lo)
        for ax in axes.ravel():
            ax.set_ylim(lo - pad, hi + pad)
        print(f"  shared y axis: {lo - pad:.1f} to {hi + pad:.1f}")

    handles = [Line2D([], [], color=GROUP_COLOR[g], lw=1.8, label=GROUP_LABEL[g])
               for g in GROUPS]
    handles += [Line2D([], [], color="0.35", lw=1.8, ls=ls, label=lbl)
                for _, ls, lbl in ARMS.values()]
    axes[0][0].legend(handles=handles, fontsize=8)

    fig.suptitle("Accuracy With One Country Substituted Into Every Question "
                 "(bands = 95% CI)", fontsize=12)
    fig.tight_layout()
    out = FIGURES / f"probe_acc_2x2_centric_distant.{args.format}"
    fig.savefig(out, bbox_inches="tight",
                dpi=200 if args.format != "pdf" else None, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
