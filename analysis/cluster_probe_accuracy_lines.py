"""Cluster probe metrics as pipeline trajectories: three panels per figure.

One figure per (benchmark, shot format, model, training arm), with x the
training stage and one line per cluster in its own colour, in three panels:

  1. default rate -- of the model's errors, the fraction whose prediction
     matches what it predicts when that cluster's representative country is
     swapped into the prompt
  2. own accuracy -- accuracy on the examples whose own country is in that
     cluster, i.e. how well the model handles that culture's questions
  3. swapped accuracy -- accuracy of the prediction made with that country
     substituted into EVERY question, scored against the original gold. How
     well you would do by assuming the whole benchmark was about one country.

Panel 2 is a slice of the examples; panels 1 and 3 use all examples with the
country replaced, excluding ones already from that country. All three come from
the same shot variant, so a figure is one prompt condition throughout.

Arms: the two aya arms share their base point, and tulu3 exists for 8b only.

    python analysis/cluster_probe_accuracy_lines.py
    python analysis/cluster_probe_accuracy_lines.py --benchmark blend
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

from analysis.cluster_probe_analysis import (CLUSTER_ORDER, CLUSTER_REPS,
                                             PROBE_SUFFIX,
                                             default_rate_among_errors,
                                             load_probe_file, probe_accuracy)
from analysis.cluster_accuracy_pipeline import CLUSTER_COLORS, cluster_acc, find

FIGURES = PROJECT_ROOT / "outputs" / "figures" / "cluster_probe_accuracy"

ORDER = ["EnglishSpeaking"] + list(CLUSTER_ORDER)
REPS = {"EnglishSpeaking": "us", **CLUSTER_REPS}

# Legend labels name the representative country, since that is what panels 1
# and 3 actually substitute into the prompt. The cluster is kept in parentheses
# because panel 2 is accuracy over EVERY country in the cluster, not over the
# representative alone -- without it, a "Netherlands" line in that panel reads
# as the Netherlands' own accuracy.
COUNTRY_NAME = {
    "us": "US", "netherlands": "Netherlands", "hungary": "Hungary",
    "bosnia_and_herzegovina": "Bosnia & Herz.", "mexico": "Mexico",
    "taiwan": "Taiwan", "philippines": "Philippines", "iran": "Iran",
}
CLUSTER_ABBR = {
    "EnglishSpeaking": "English", "ProtestantEurope": "Protestant Eur.",
    "CatholicEurope": "Catholic Eur.", "Orthodox": "Orthodox",
    "LatinAmerica": "Latin America", "Confucian": "Confucian",
    "SouthAsia": "South Asia", "AfricanIslamic": "African-Islamic",
}


def label_for(cl: str) -> str:
    return f"{COUNTRY_NAME[REPS[cl]]} ({CLUSTER_ABBR[cl]})"

BENCHES = ["normad", "culturalbench", "blend"]

# arm -> (x labels, conditions). The aya arms start from the shared base point.
ARMS = {
    "aya_cult":   (["Base", "SFT", "SFT+DPO"],
                   ["base", "sft_aya_cult", "sftdpo_aya_cult"]),
    "aya_nocult": (["Base", "SFT", "SFT+DPO"],
                   ["base", "sft_aya_nocult", "sftdpo_aya_nocult"]),
    "tulu3":      (["Base", "SFT", "DPO"],
                   ["base", "tulu3_sft", "tulu3_dpo"]),
}

ARM_LABEL = {"aya_cult": "Cultural Data", "aya_nocult": "Generic Data",
             "tulu3": "Tulu-3"}
MODEL_LABEL = {"8b": "Llama-3.1-8B", "gemma4": "Gemma-4-12B"}
BENCH_TITLE = {"normad": "NormAd", "blend": "BLEnD",
               "culturalbench": "CulturalBench"}


PANELS = [
    ("Default Rate Among Errors (%)", 0),
    ("Accuracy on All Questions From That Cluster (%)", 1),
    ("Accuracy With That Country in Every Question (%)", 2),
]
# key for --only-panel, which writes a single-axes figure instead of the trio
PANEL_KEY = {"default": 0, "own": 1, "swapped": 2}


def series(bench: str, model: str, conds: list[str], min_n: int):
    """{cluster: (cdr, own_acc, swapped_acc)} as percent lists per stage."""
    sfx = PROBE_SUFFIX[bench]
    out = {}
    for cl in ORDER:
        cdr, own, swapped = [], [], []
        for cond in conds:
            pr = load_probe_file(cond, model, REPS[cl], benchmark=bench)
            cdr.append(100 * default_rate_among_errors(pr, REPS[cl])[0]
                       if pr else None)
            swapped.append(100 * probe_accuracy(pr, REPS[cl]) if pr else None)
            a = cluster_acc(find(bench, cond, model, sfx))
            cell = a.get(cl) if a else None
            own.append(cell[0] if cell and cell[1] >= min_n else None)
        if any(v is not None for v in cdr):
            out[cl] = (cdr, own, swapped)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", nargs="*", default=BENCHES, choices=BENCHES)
    ap.add_argument("--model", nargs="*", default=["8b", "gemma4"],
                    choices=["8b", "gemma4"])
    ap.add_argument("--min-n", type=int, default=30,
                    help="Omit a cluster's accuracy line below this many examples.")
    ap.add_argument("--only-panel", default=None, choices=list(PANEL_KEY),
                    help="Write a single-panel figure for just this metric "
                         "instead of all three side by side.")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "jpg"],
                    help="Output image format. png/jpg render at 200 dpi, for "
                         "pasting into slides or a thread.")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for bench in args.benchmark:
        sfx = PROBE_SUFFIX[bench]
        for model in args.model:
            for arm, (xlabels, conds) in ARMS.items():
                # no Gemma Tulu-3 release
                if arm == "tulu3" and model != "8b":
                    continue
                data = series(bench, model, conds, args.min_n)
                if not data or any(v is None for c in data.values() for v in c[0]):
                    missing = [c for c, v in data.items() if None in v[0]]
                    print(f"  {bench}{sfx} {model} {arm}: incomplete"
                          + (f" ({', '.join(missing)})" if missing else ""))
                    if not data:
                        continue

                if args.only_panel:
                    idx = PANEL_KEY[args.only_panel]
                    title = PANELS[idx][0]
                    fig, ax = plt.subplots(figsize=(7.2, 5.0))
                    panels = [(ax, title, idx)]
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
                    panels = [(a, t, i) for a, (t, i) in zip(axs, PANELS)]

                xs = range(len(xlabels))
                for ax, title, idx in panels:
                    for cl, vals in data.items():
                        ys = vals[idx]
                        if any(v is None for v in ys):
                            continue
                        ax.plot(xs, ys, color=CLUSTER_COLORS[cl], marker="o",
                                markersize=5, linewidth=1.8, label=label_for(cl))
                    ax.set_xticks(list(xs))
                    ax.set_xticklabels(xlabels)
                    ax.set_xlim(-0.25, len(xlabels) - 0.75)
                    ax.set_xlabel("Training Stage")
                    ax.set_ylabel(title)
                    if not args.only_panel:
                        ax.set_title(title, fontsize=10)
                    ax.grid(alpha=0.25, linewidth=0.5, axis="y")

                handles = [Line2D([], [], color=CLUSTER_COLORS[cl], lw=1.8,
                                  marker="o", markersize=5, label=label_for(cl))
                           for cl in data]
                panels[0][0].legend(handles=handles, fontsize=8, ncol=2,
                                    title="Probe Country", title_fontsize=8)
                head = (f"{BENCH_TITLE[bench]} — {MODEL_LABEL[model]}, "
                        f"{ARM_LABEL[arm]}")
                if args.only_panel:
                    panels[0][0].set_title(head, fontsize=12)
                else:
                    fig.suptitle(head, fontsize=11)
                fig.tight_layout()
                tag = f"_{args.only_panel}" if args.only_panel else ""
                out = (FIGURES /
                       f"probe_acc_{bench}{sfx}_{model}_{arm}{tag}.{args.format}")
                fig.savefig(out, bbox_inches="tight",
                            dpi=200 if args.format != "pdf" else None,
                            facecolor="white")
                plt.close(fig)
                wrote += 1
                print(f"  {bench}{sfx:9s} {model:7s} {arm:11s} "
                      f"{len(data)} clusters -> {out.name}")
    print(f"\n{wrote} figure(s) in {FIGURES}")


if __name__ == "__main__":
    main()
