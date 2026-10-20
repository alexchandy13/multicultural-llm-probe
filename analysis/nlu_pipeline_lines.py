"""General-NLU accuracy trajectory through the training pipeline, one line per task.

Companion to analysis/cluster_accuracy_pipeline.py with the same axes: x is the
training stage, one line per series, colour fixed across figures. There are no
cultures to split on here, so a line is an NLU task instead of an IW cluster —
the figure answers whether a stage that moved cultural accuracy also moved
general capability, or left it flat.

--shot picks the in-context setting. Note _nfs in a filename means NEUTRAL
few-shot -- two country-free demonstrations that fix the answer format without
supplying domain knowledge -- not zero-shot; true zero-shot runs carry no shot
suffix. Both settings are available for every NLU cell, so both can be plotted.

Three pipelines, selected with --pipeline:
  aya    base -> SFT -> SFTDPO, solid = cult arm, dotted = nocult arm
  tulu3  base -> SFT -> DPO  (8b only, single arm)
  olmo3  base -> SFT -> DPO; solid = plain prompt, dotted = with a neutral
         system prompt, which is the prompting control from the OLMo 3 audit
         rather than a second training arm.

    python analysis/nlu_pipeline_lines.py
    python analysis/nlu_pipeline_lines.py --shot zero --pipeline olmo3
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
FIGURES = PROJECT_ROOT / "outputs" / "figures" / "nlu_pipeline"

TASKS = ["boolq", "csqa", "mrpc", "qnli"]
TASK_LABELS = {"boolq": "BoolQ", "csqa": "CSQA", "mrpc": "MRPC", "qnli": "QNLI"}
TASK_COLORS = {
    "boolq": "#1f78b4",
    "csqa":  "#33a02c",
    "mrpc":  "#e6ab02",
    "qnli":  "#6a3d9a",
}
ARM_STYLES = {"cult": "-", "nocult": ":", "plain": "-", "sys": ":", "only": "-"}
ARM_LABELS = {"cult": "cult", "nocult": "nocult",
              "plain": "plain prompt", "sys": "system prompt"}

# stem(task, stage) -> filename stem, so each pipeline can own its naming.
PIPELINES = {
    "aya": {
        "sizes": ["8b", "gemma4"],
        "xlabels": ["base", "SFT", "SFTDPO"],
        "arms": {
            "cult":   ["base", "sft_aya_cult", "sftdpo_aya_cult"],
            "nocult": ["base", "sft_aya_nocult", "sftdpo_aya_nocult"],
        },
        "path": lambda t, cond, size, sfx: f"nlu_{t}_{cond}_{size}{sfx}.json",
    },
    "tulu3": {
        "sizes": ["8b"],
        "xlabels": ["base", "SFT", "DPO"],
        "arms": {"only": ["base", "tulu3_sft", "tulu3_dpo"]},
        "path": lambda t, cond, size, sfx: f"nlu_{t}_{cond}_{size}{sfx}.json",
    },
    "olmo3": {
        "sizes": ["olmo3"],
        "xlabels": ["base", "SFT", "DPO"],
        "arms": {
            "plain": ["olmo3", "olmo3_sft", "olmo3_dpo"],
            # the base model has no chat template, so no system prompt exists
            # for it; the sys arm starts from the same shared base point.
            "sys":   ["olmo3", "olmo3_sft_sys", "olmo3_dpo_sys"],
        },
        # the stage lives in the model slot here, not the condition slot
        "path": lambda t, cond, size, sfx: (
            f"nlu_{t}_base_{cond[:-4]}{sfx}_sys.json" if cond.endswith("_sys")
            else f"nlu_{t}_base_{cond}{sfx}.json"),
    },
}


def accuracy(path: Path) -> tuple[float, float] | None:
    """(accuracy %, 1.96 * binomial SE in points), or None if the run is missing."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    p, n = d["accuracy_overall"], d["n"]
    return 100 * p, 100 * 1.96 * sqrt(p * (1 - p) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="aya", choices=list(PIPELINES))
    ap.add_argument("--shot", default="nfs", choices=["nfs", "zero"],
                    help="nfs = 2 country-free neutral shots (the _nfs files); "
                         "zero = no in-context examples (bare filenames).")
    args = ap.parse_args()

    spec = PIPELINES[args.pipeline]
    sfx = "_nfs" if args.shot == "nfs" else ""
    xlabels, arms = spec["xlabels"], spec["arms"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    wrote = 0

    for size in spec["sizes"]:
        # {task: {arm: [(acc, ci) per stage]}}
        data: dict[str, dict[str, list]] = {}
        for task in TASKS:
            per_arm = {}
            for arm, conds in arms.items():
                pts = [accuracy(BEHAVIORAL / spec["path"](task, c, size, sfx))
                       for c in conds]
                if any(p is None for p in pts):
                    missing = [c for c, p in zip(conds, pts) if p is None]
                    print(f"  {size} {task} {arm}: missing {', '.join(missing)}")
                    continue
                per_arm[arm] = pts
            if per_arm:
                data[task] = per_arm
        if not data:
            print(f"  {size}: nothing to plot")
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        xs = list(range(len(xlabels)))
        for task, per_arm in data.items():
            for arm, pts in per_arm.items():
                ax.errorbar(xs, [a for a, _ in pts], yerr=[c for _, c in pts],
                            color=TASK_COLORS[task], linestyle=ARM_STYLES[arm],
                            marker="o", markersize=4, linewidth=1.8,
                            elinewidth=0.8, capsize=2,
                            label=TASK_LABELS[task] if arm in ("cult", "plain",
                                                               "only") else None)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.25, len(xlabels) - 0.75)
        ax.set_xlabel("training stage")
        ax.set_ylabel("accuracy (%)")
        styles = " ".join(f"{ARM_STYLES[a]} = {ARM_LABELS[a]}," for a in arms
                          if a in ARM_LABELS).rstrip(",")
        ax.set_title(f"General NLU through the {args.pipeline} pipeline — {size}, "
                     f"{args.shot}-shot"
                     + (f"\n{styles}" if styles else ""))
        ax.grid(alpha=0.25, linewidth=0.5, axis="y")
        ax.legend(fontsize=8, title="task", title_fontsize=8)
        fig.tight_layout()
        out = FIGURES / f"nlu_pipeline_{args.pipeline}_{args.shot}shot_{size}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        wrote += 1

        print(f"\n### {args.pipeline} {size} {args.shot}-shot   "
              + " -> ".join(xlabels) + "   (bars = 95% binomial CI)")
        for task, per_arm in data.items():
            for arm, pts in per_arm.items():
                cells = "  ".join(f"{a:5.1f}±{c:.1f}" for a, c in pts)
                print(f"  {TASK_LABELS[task]:6s} {arm:7s} {cells}"
                      f"   Δend {pts[-1][0] - pts[0][0]:+5.1f}")

    print(f"\n{wrote} figure(s) in {FIGURES}")


if __name__ == "__main__":
    main()
