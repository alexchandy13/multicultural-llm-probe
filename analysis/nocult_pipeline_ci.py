"""Non-cultural (nocult) training pipeline, with translucent confidence bands.

Three files, each Base -> SFT -> SFT+DPO on the nocult arm only:

  nlu     one panel, BoolQ and CSQA together -- colour is the model,
          marker shape is the task (circle BoolQ, triangle CSQA)
  blend   two panels, one per model, US-centric vs US-distant clusters
  normad  two panels, one per model, US-centric vs US-distant clusters

--partition picks how Western / Non-Western is defined:

  group (default)  the `group` field written into each eval output, which is
                   what the submitted paper used: Western / Non-Western /
                   Other, with Other excluded. NormAd then reads the
                   _nfs_mpw_usprobe_batch_calibrated files and BLEnD the
                   _nfs_usprobe ones, so these figures match the paper's
                   tables exactly.
  iw               not in / in US_SIMILAR_CLUSTERS from culnig.clusters
                   (English-speaking plus Protestant Europe), which uses every
                   country rather than dropping Other, and reads NormAd
                   _fs2_yn.

The two are NOT interchangeable: a third of NormAd (616 of 1,818) and 8,200 of
BLEnD's 23,032 sit in "Other" and are absent from the paper's partition.

Bands are 95% binomial intervals, 1.96*sqrt(p(1-p)/n).

NLU reads the _nfs runs (two country-free neutral shots); NormAd uses _fs2_yn
and BLEnD _nfs, matching the rest of the analysis.

    python analysis/nocult_pipeline_ci.py
    python analysis/nocult_pipeline_ci.py --format pdf
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
from matplotlib.lines import Line2D

from analysis.cluster_accuracy_pipeline import BEHAVIORAL, cluster_acc, find
from culnig.clusters import US_SIMILAR_CLUSTERS, cluster_of

FIGURES = PROJECT_ROOT / "outputs" / "figures" / "nocult_pipeline"

CONDS = ["base", "sft_aya_nocult", "sftdpo_aya_nocult"]

# Per-partition file variants. The paper's NormAd numbers are batch-calibrated
# neutral-few-shot multi-prompt-word runs; its BLEnD numbers are the plain
# neutral-few-shot ones.
PAPER_SUFFIX = {"normad": "_nfs_mpw_usprobe_batch_calibrated",
                "blend": "_nfs_usprobe",
                "culturalbench": "_nfs4_usprobe"}
IW_SUFFIX = {"normad": "_fs2_yn", "blend": "_nfs", "culturalbench": "_nfs4"}


def find_for(bench: str, cond: str, size: str, partition: str):
    """Resolve the prediction file for this benchmark under this partition."""
    if partition == "group":
        return BEHAVIORAL / f"{bench}_{cond}_{size}{PAPER_SUFFIX[bench]}.json"
    return find(bench, cond, size, IW_SUFFIX[bench])


def group_of(x: dict, partition: str) -> str | None:
    """Map one prediction to us_centric / us_distant, or None to drop it."""
    if partition == "group":
        g = x.get("group")
        return {"Western": "us_centric",
                "Non-Western": "us_distant"}.get(g)   # "Other" -> None
    cl = cluster_of(x["country"])
    if cl is None:
        return None
    return "us_centric" if cl in US_SIMILAR_CLUSTERS else "us_distant"
XLABELS = ["Base", "SFT", "SFT+DPO"]
MODELS = ["8b", "gemma4"]
MODEL_LABEL = {"8b": "Llama-3.1-8B", "gemma4": "Gemma-4-12B"}
TASK_LABEL = {"boolq": "BoolQ", "csqa": "CSQA"}
TASK_MARKER = {"boolq": "o", "csqa": "^"}
MODEL_COLOR = {"8b": "#b2182b", "gemma4": "#2166ac"}

BENCH_LABEL = {"blend": "BLEnD", "normad": "NormAd",
               "culturalbench": "CulturalBench"}
GROUPS = ["us_centric", "us_distant"]
GROUP_LABEL = {"us_centric": "US-Centric", "us_distant": "US-Distant"}
GROUP_COLOR = {"us_centric": "#4477AA", "us_distant": "#CC6677"}


def two_group_acc(path, partition: str = "group"):
    """{group: (accuracy%, n)}, pooled over examples.

    Pooled rather than averaged over per-cluster rates, so a group with 7,000
    examples counts 50x one with 140 -- which is what the binomial band then
    describes. US-anchored rows are kept here (accuracy on US questions is
    meaningful); they are dropped only for the counterfactual metrics.
    """
    if path is None or not path.exists():
        return None
    pr = json.loads(path.read_text())["predictions"]
    out = {}
    for g in GROUPS:
        sub = [x for x in pr if group_of(x, partition) == g]
        if sub:
            out[g] = (100 * sum(x["pred"] == x["gold"] for x in sub) / len(sub),
                      len(sub))
    return out


def ci(p_pct: float, n: int) -> float:
    """95% binomial half-width in percentage points."""
    p = p_pct / 100
    return 100 * 1.96 * sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def set_span(ax, lo: float, hi: float, span: float) -> None:
    """Give every panel the same y-axis LENGTH, centred on its own data.

    A shared 0-100 axis would flatten every line; a per-panel autoscale makes a
    1-point move look like a 10-point one. Fixing the span instead keeps slopes
    comparable across panels while each stays readable. If a panel's data plus
    bands exceed the span, it widens rather than clipping, and says so.
    """
    need = hi - lo
    if need > span:
        pad = 0.04 * need
        ax.set_ylim(lo - pad, hi + pad)
        print(f"    note: {ax.get_title()} spans {need:.1f} pts, wider than "
              f"{span:g} — axis widened")
        return
    mid = (lo + hi) / 2
    ax.set_ylim(mid - span / 2, mid + span / 2)


def band(ax, xs, ys, half, color, label=None, ls="-", marker="o",
         markersize=5):
    ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
            markersize=markersize, linewidth=1.8, label=label)
    ax.fill_between(xs, [y - h for y, h in zip(ys, half)],
                    [y + h for y, h in zip(ys, half)],
                    color=color, alpha=0.15, linewidth=0)


def nlu_point(task: str, cond: str, size: str):
    p = BEHAVIORAL / f"nlu_{task}_{cond}_{size}_nfs.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    acc = 100 * d["accuracy_overall"]
    return acc, ci(acc, d["n"])


def fig_nlu(fmt: str, span: float) -> Path | None:
    """One panel: colour distinguishes the model, marker shape the task."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    xs = range(len(XLABELS))
    lo, hi = [], []
    for task in ("boolq", "csqa"):
        for size in MODELS:
            pts = [nlu_point(task, c, size) for c in CONDS]
            if any(v is None for v in pts):
                continue
            band(ax, xs, [v[0] for v in pts], [v[1] for v in pts],
                 MODEL_COLOR[size], marker=TASK_MARKER[task], markersize=9)
            lo += [a - h for a, h in pts]
            hi += [a + h for a, h in pts]
    if not lo:
        plt.close(fig)
        return None
    ax.set_xticks(list(xs))
    ax.set_xticklabels(XLABELS)
    ax.set_xlim(-0.25, len(XLABELS) - 0.75)
    ax.set_xlabel("Training Stage")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25, linewidth=0.5, axis="y")
    # two separate keys: one for colour (model), one for marker (task), so the
    # legend stays four entries rather than one per model-task combination
    handles = [Line2D([], [], color=MODEL_COLOR[m], lw=1.8,
                      label=MODEL_LABEL[m]) for m in MODELS]
    handles += [Line2D([], [], color="0.35", lw=1.8, marker=TASK_MARKER[t],
                       markersize=9, label=TASK_LABEL[t])
                for t in ("boolq", "csqa")]
    ax.legend(handles=handles, fontsize=8)
    set_span(ax, min(lo), max(hi), span)
    fig.suptitle("General NLU Through Non-Cultural Post-Training "
                 "(bands = 95% CI)", fontsize=11)
    fig.tight_layout()
    out = FIGURES / f"nocult_pipeline_nlu.{fmt}"
    fig.savefig(out, bbox_inches="tight",
                dpi=200 if fmt != "pdf" else None, facecolor="white")
    plt.close(fig)
    return out


def fig_clustered(bench: str, mid: str, fmt: str, min_n: int,
                  span: float, partition: str = "group") -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    xs = range(len(XLABELS))
    any_data = False
    for ax, size in zip(axes, MODELS):
        stages = [two_group_acc(find_for(bench, c, size, partition),
                                partition) for c in CONDS]
        if any(s is None for s in stages):
            ax.set_title(f"{MODEL_LABEL[size]} — no data", fontsize=11)
            continue
        lo, hi = [], []
        for g in GROUPS:
            if not all(g in s and s[g][1] >= min_n for s in stages):
                continue
            ys = [s[g][0] for s in stages]
            half = [ci(s[g][0], s[g][1]) for s in stages]
            band(ax, xs, ys, half, GROUP_COLOR[g],
                 label=GROUP_LABEL[g])
            lo += [y - h for y, h in zip(ys, half)]
            hi += [y + h for y, h in zip(ys, half)]
            any_data = True
        ax.set_xticks(list(xs)); ax.set_xticklabels(XLABELS)
        ax.set_xlim(-0.25, len(XLABELS) - 0.75)
        ax.set_xlabel("Training Stage"); ax.set_ylabel("Accuracy (%)")
        ax.set_title(MODEL_LABEL[size], fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5, axis="y")
        ax.legend(fontsize=8)
        if lo:
            set_span(ax, min(lo), max(hi), span)
    if not any_data:
        plt.close(fig); return None
    fig.suptitle(f"{BENCH_LABEL[bench]}: US-Centric vs US-Distant "
                 f"Through Non-Cultural Post-Training (bands = 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES / f"nocult_pipeline_{bench}.{fmt}"
    fig.savefig(out, bbox_inches="tight",
                dpi=200 if fmt != "pdf" else None, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="png", choices=["png", "pdf", "jpg"])
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--partition", default="group", choices=["group", "iw"],
                    help="group = the paper's Western/Non-Western field; "
                         "iw = Inglehart-Welzel US_SIMILAR_CLUSTERS.")
    ap.add_argument("--y-span", type=float, default=25.0,
                    help="Percentage-point length of every y axis, so slopes "
                         "are comparable across panels. Each panel centres this "
                         "window on its own data.")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    for out in (fig_nlu(args.format, args.y_span),
                fig_clustered("blend", "", args.format, args.min_n,
                              args.y_span, args.partition),
                fig_clustered("normad", "", args.format, args.min_n,
                              args.y_span, args.partition)):
        print(f"  wrote {out.name}" if out else "  (skipped: no data)")
    print(f"\nin {FIGURES}")


if __name__ == "__main__":
    main()
