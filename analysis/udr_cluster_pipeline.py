"""US-default rate by IW cluster, as a trajectory through the training pipeline.

Same axes as analysis/cluster_accuracy_pipeline.py -- x is the training stage,
one line per cluster, colour fixed per cluster, solid = cult arm and dotted =
nocult -- but the y value is a US-default statistic computed on the subset of
questions whose country belongs to that cluster.

--metric picks what is plotted:
  udr     P(pred == us_pred | pred != gold), the raw rate
  kappa   that rate corrected for how often the two would agree given the
          model's own label frequencies; see analysis/udr_pipeline.py
  flip    P(pred != us_pred) over ALL questions in the cluster, no error
          conditioning and so no selection effect

udr alone is not safe to read across stages: a checkpoint that collapses onto
one label agrees with its own counterfactual for free, so plot kappa beside it.

US-anchored questions are dropped everywhere, which also means the
EnglishSpeaking line is the non-US English-speaking countries.

    python analysis/udr_cluster_pipeline.py --metric kappa
    python analysis/udr_cluster_pipeline.py --metric udr --pipeline olmo3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from culnig.clusters import CLUSTERS, cluster_of
from analysis.cluster_accuracy_pipeline import (ARM_STYLES, BENCH_BY_SHOT,
                                                CLUSTER_COLORS, PIPELINES, find)
from analysis.udr_pipeline import is_us

FIGURES = PROJECT_ROOT / "outputs" / "figures" / "udr_cluster_pipeline"

METRIC_LABELS = {
    "udr":   "US-default rate among errors (%)",
    "kappa": "US-default agreement above chance (kappa)",
    "flip":  "answers changed by the US swap, all questions (%)",
}


def cluster_udr(path: Path | None):
    """{cluster: (udr%, chance%, kappa, flip%, n_err)} over non-US questions."""
    if path is None:
        return None
    pr = [x for x in json.load(open(path))["predictions"]
          if x.get("us_pred") is not None and not is_us(x.get("country", ""))]
    out = {}
    for cl in CLUSTERS:
        sub = [x for x in pr if cluster_of(x["country"]) == cl]
        if not sub:
            continue
        flip = 100 * sum(x["pred"] != x["us_pred"] for x in sub) / len(sub)
        err = [x for x in sub if x["pred"] != x["gold"]]
        if not err:
            continue
        agree = sum(x["pred"] == x["us_pred"] for x in err) / len(err)
        p, u = Counter(x["pred"] for x in err), Counter(x["us_pred"] for x in err)
        chance = sum(p[c] * u[c] for c in set(p) | set(u)) / len(err) ** 2
        kappa = (agree - chance) / (1 - chance) if chance < 1 else float("nan")
        out[cl] = (100 * agree, 100 * chance, kappa, flip, len(err))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="kappa", choices=list(METRIC_LABELS))
    ap.add_argument("--shot", default="fs2", choices=list(BENCH_BY_SHOT))
    ap.add_argument("--pipeline", default="aya", choices=list(PIPELINES))
    ap.add_argument("--min-err", type=int, default=40,
                    help="Drop clusters with fewer errors than this in any "
                         "stage. NormAd has only ~250 errors in total, so most "
                         "of its clusters cannot support a per-cluster rate.")
    ap.add_argument("--quiet", action="store_true", help="Figures only.")
    args = ap.parse_args()

    col = {"udr": 0, "kappa": 2, "flip": 3}[args.metric]
    spec = PIPELINES[args.pipeline]
    xlabels, arms = spec["xlabels"], spec["arms"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    wrote = 0

    for size in spec["sizes"]:
        for bench, mid in BENCH_BY_SHOT[args.shot]:
            data = {}
            for arm, stages in arms.items():
                vals = [cluster_udr(find(bench, s, size, mid)
                                    if spec["slot"] == "cond"
                                    else find(bench, "base", s, mid))
                        for s in stages]
                if not any(v is None for v in vals):
                    data[arm] = vals
            if not data:
                continue
            keep = [c for c in CLUSTERS
                    if all(c in v and v[c][4] >= args.min_err
                           for vs in data.values() for v in vs)]
            if not keep:
                print(f"  {size} {bench}{mid}: no cluster reaches "
                      f"{args.min_err} errors")
                continue

            fig, ax = plt.subplots(figsize=(7.5, 5))
            xs = range(len(xlabels))
            labelled = next(iter(data))
            for cl in keep:
                for arm, vs in data.items():
                    ax.plot(xs, [v[cl][col] for v in vs],
                            color=CLUSTER_COLORS[cl], linestyle=ARM_STYLES[arm],
                            marker="o", markersize=4, linewidth=1.8,
                            label=cl if arm == labelled else None)
            ax.set_xticks(list(xs)); ax.set_xticklabels(xlabels)
            ax.set_xlabel("training stage")
            ax.set_ylabel(METRIC_LABELS[args.metric])
            ax.set_title(f"US-defaulting by cluster — {bench}{mid} {size}"
                         + (f"\nsolid = cult, dotted = nocult"
                            if len(data) > 1 else ""))
            ax.grid(alpha=0.25, linewidth=0.5, axis="y")
            ax.legend(fontsize=8, title="IW cluster", title_fontsize=8, ncol=2)
            fig.tight_layout()
            out = (FIGURES / f"udr_{args.metric}_{args.pipeline}_"
                             f"{bench}{mid}_{size}.pdf")
            fig.savefig(out, bbox_inches="tight"); plt.close(fig)
            wrote += 1

            if not args.quiet:
                print(f"\n### {bench}{mid} {size} ({args.pipeline}) — "
                      f"{args.metric}, non-US questions, min {args.min_err} errors")
                hdr = f"{'cluster':18s} {'n_err':>6s}"
                for arm in data:
                    hdr += "  " + " ".join(f"{x[:6]:>6s}" for x in xlabels) \
                           + f" [{arm}]"
                print(hdr)
                for cl in keep:
                    row = f"{cl:18s} {data[labelled][0][cl][4]:6d}"
                    for arm, vs in data.items():
                        fmt = "{:6.3f}" if args.metric == "kappa" else "{:6.1f}"
                        row += "  " + " ".join(fmt.format(v[cl][col]) for v in vs) \
                               + " " * (len(arm) + 3)
                    print(row)
    print(f"\n{wrote} figure(s) in {FIGURES}")


if __name__ == "__main__":
    main()
