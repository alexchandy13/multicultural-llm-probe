"""Accuracy trajectory through the training pipeline, one line per IW cluster.

x-axis is the stage (base -> SFT -> SFTDPO), so each line is one cluster's path
through post-training. Colour encodes the cluster and is fixed across figures;
solid is the cult arm and dotted the nocult arm, both starting from the shared
base point. A cluster whose dotted line falls below its solid one is a cluster
the cultural training data helped.

Accuracy is example-weighted within a cluster, not the mean of per-country
accuracies — clusters have very uneven country counts (NormAd: ProtestantEurope
55 examples, SouthAsia 412), so averaging per-country rates would overweight the
small ones.

    python analysis/cluster_accuracy_pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from culnig.clusters import CLUSTERS, cluster_of

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES = PROJECT_ROOT / "outputs" / "figures" / "cluster_accuracy_pipeline"

# Fixed per cluster so colour means the same thing in every figure. Warm for
# the Western-adjacent clusters, cool and green for the rest, matching
# analysis/cluster_specific_layer_lines.py.
CLUSTER_COLORS = {
    # English-speaking (the US probe) takes blue, swapped with South Asia, so
    # the reference country reads as the neutral/baseline line rather than the
    # alarm-coloured one.
    "EnglishSpeaking":  "#4393c3",
    "ProtestantEurope": "#ef8a62",
    "CatholicEurope":   "#f4a582",
    "Orthodox":         "#998ec3",
    "Confucian":        "#2166ac",
    "SouthAsia":        "#b2182b",
    "AfricanIslamic":   "#1b7837",
    "LatinAmerica":     "#762a83",
}

# One entry per training pipeline. "arms" maps an arm to its stage sequence,
# base first and shared between arms. "slot" says which part of the filename the
# stage name goes into: the aya and tulu3 runs vary the condition at a fixed
# model size, while the OLMo 3 checkpoints are separate models, so there the
# condition is pinned to "base" and the stage occupies the model slot.
PIPELINES = {
    "aya": {
        "sizes": ["8b", "gemma4"], "slot": "cond",
        "xlabels": ["base", "SFT", "SFTDPO"],
        "arms": {"cult":   ["base", "sft_aya_cult", "sftdpo_aya_cult"],
                 "nocult": ["base", "sft_aya_nocult", "sftdpo_aya_nocult"]},
    },
    "tulu3": {
        "sizes": ["8b"], "slot": "cond",
        "xlabels": ["base", "SFT", "DPO"],
        "arms": {"cult": ["base", "tulu3_sft", "tulu3_dpo"]},
    },
    "olmo3": {
        "sizes": ["olmo3"], "slot": "size",
        "xlabels": ["base", "SFT", "DPO"],
        # the sys arm is the neutral-system-prompt control from the OLMo 3
        # audit, not a second training arm. The base model has no chat template
        # and so no system prompt, so it shares the plain base point.
        "arms": {"plain": ["olmo3", "olmo3_sft", "olmo3_dpo"],
                 "sys":   ["olmo3", "olmo3_sft_sys", "olmo3_dpo_sys"]},
    },
}
ARM_STYLES = {"cult": "-", "nocult": ":", "plain": "-", "sys": ":"}
# Shot settings, which live in the filename suffix. Note _nfs is NEUTRAL
# few-shot (2 country-free demonstrations; _nfs4 is 4 of them), not zero-shot --
# see eval_culturalbench.build_neutral_fewshot_prefix. True zero-shot comes from
# --no-fewshot / omitting --neutral-fewshot and carries no shot suffix at all.
# NormAd also has two prompt formats, yn (plain yes/no) and mpw
# (multi-prompt-word, the standing eval default), so both are listed where they
# exist. NormAd with no format suffix at all is the older 3-way label set, a
# different task (base 8b scores 36.7 there vs 61.1 on yn), so it is not a
# zero-shot counterpart to anything here and is left out.
BENCH_BY_SHOT = {
    "fs2":  [("normad", "_fs2_yn"), ("culturalbench", "_nfs4"), ("blend", "_nfs")],
    "nfs":  [("normad", "_nfs_yn"), ("normad", "_nfs_mpw"),
             ("culturalbench", "_nfs4"), ("blend", "_nfs")],
    "zero": [("normad", "_yn"), ("culturalbench", ""), ("blend", "")],
}


def find(bench: str, cond: str, size: str, mid: str) -> Path | None:
    """Locate a run, tolerating the pre-_usprobe filename convention.

    The earlier zero-shot runs were written before the _usprobe suffix existed,
    but they already carried the swapped-country prediction and the per-example
    country, so they are the same kind of file under an older name. Where both
    spellings exist they are separate runs of the same config and differ by
    under a point (NormAd base 8b yn: 61.11 vs 60.46), so the suffixed one wins.
    """
    sysp = ""
    if size.endswith("_sys"):            # system-prompt control run
        size, sysp = size[:-4], "_sys"
    for name in (f"{bench}_{cond}_{size}{mid}_usprobe{sysp}.json",
                 f"{bench}_{cond}_{size}{mid}{sysp}.json"):
        path = BEHAVIORAL / name
        if path.exists():
            return path
    return None


def cluster_acc(path: Path | None):
    """{cluster: (accuracy%, n)} example-weighted."""
    if path is None or not path.exists():
        return None
    pr = json.load(open(path))["predictions"]
    out = {}
    for cl in CLUSTERS:
        sub = [x for x in pr if cluster_of(x["country"]) == cl]
        if sub:
            out[cl] = (100*sum(x["pred"] == x["gold"] for x in sub)/len(sub), len(sub))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="aya", choices=list(PIPELINES))
    ap.add_argument("--shot", default="fs2", choices=list(BENCH_BY_SHOT),
                    help="fs2 = 2 real in-context examples (NormAd only; the "
                         "other two are neutral few-shot either way); nfs = 2 "
                         "country-free neutral shots; zero = no shots at all. "
                         "Coverage is incomplete for zero.")
    ap.add_argument("--min-n", type=int, default=30,
                    help="Drop clusters with fewer examples than this; at n<30 a "
                         "single flipped example moves accuracy by >3 points.")
    args = ap.parse_args()

    spec = PIPELINES[args.pipeline]
    xlabels, arms = spec["xlabels"], spec["arms"]
    bench_specs = BENCH_BY_SHOT[args.shot]
    FIGURES.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for size in spec["sizes"]:
        for bench, mid in bench_specs:
            # {arm: [per-stage {cluster: (acc, n)}]}
            data = {}
            for arm, conds in arms.items():
                stages = [cluster_acc(find(bench, c, size, mid)
                                      if spec["slot"] == "cond"
                                      else find(bench, "base", c, mid))
                          for c in conds]
                if any(st is None for st in stages):
                    gaps = [c for c, st in zip(conds, stages) if st is None]
                    print(f"  {size} {bench}{mid} {arm}: no run for "
                          f"{', '.join(gaps)}")
                    continue
                data[arm] = stages
            if not data:
                print(f"  {size} {bench}{mid}: missing conditions"); continue

            keep = [c for c in CLUSTERS
                    if all(c in st and st[c][1] >= args.min_n
                           for stages in data.values() for st in stages)]
            if not keep:
                print(f"  {size} {bench}{mid}: no cluster clears --min-n"); continue

            fig, ax = plt.subplots(figsize=(7.5, 5))
            xs = range(len(xlabels))
            # label whichever arm actually ran, not always cult -- some
            # zero-shot grids have only the nocult arm
            labelled = next(iter(data))
            for cl in keep:
                for arm, stages in data.items():
                    ax.plot(xs, [st[cl][0] for st in stages],
                            color=CLUSTER_COLORS[cl], linestyle=ARM_STYLES[arm],
                            marker="o", markersize=4, linewidth=1.8,
                            label=cl if arm == labelled else None)
            ax.set_xticks(list(xs)); ax.set_xticklabels(xlabels)
            ax.set_xlabel("training stage")
            ax.set_ylabel("accuracy (%)")
            ax.set_title(f"Accuracy through the {args.pipeline} pipeline — "
                         f"{bench}{mid} {size}\n"
                         f"solid = cult, dotted = nocult"
                         if len(data) > 1 else
                         f"Accuracy through the {args.pipeline} pipeline — "
                         f"{bench}{mid} {size}")
            ax.grid(alpha=0.25, linewidth=0.5, axis="y")
            ax.legend(fontsize=8, title="IW cluster", title_fontsize=8, ncol=2)
            fig.tight_layout()
            out = (FIGURES /
                   f"pipeline_accuracy_{args.pipeline}_{bench}{mid}_{size}.pdf")
            fig.savefig(out, bbox_inches="tight"); plt.close(fig)
            wrote += 1
            print(f"  {size:7s} {bench + mid:22s} {len(keep)} clusters, "
                  f"{len(data)} arm(s) -> {out.name}")
    print(f"\n{wrote} figure(s) in {FIGURES}")


if __name__ == "__main__":
    main()
