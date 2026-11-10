"""Bar charts: how often the country-free prediction matches each probe country.

One chart per model/condition, eight bars -- the eight IW-cluster
representatives. A tall bar means that naming that country gives the same
answers as naming no country at all, so the country-free default looks like
that country's answers. If all eight are level the model is simply ignoring
the country; one country standing clear of the rest is a genuine default.

Bars stay in fixed cluster order rather than sorted by height, so the three
panels can be read against each other -- sorting would move a country between
panels and make the comparison the chart exists for impossible.

Colours and labels come from the existing cluster palette
(analysis/cluster_accuracy_pipeline.py, analysis/cluster_probe_accuracy_lines.py)
so these figures match the rest of the paper's; US is blue there and stays blue
here.

    python analysis/nocountry_agreement_bars.py --benchmark normad
    python analysis/nocountry_agreement_bars.py --benchmark blend
    python analysis/nocountry_agreement_bars.py --benchmark normad --combined
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.cluster_accuracy_pipeline import CLUSTER_COLORS
from analysis.nocountry_agreement import (
    BENCHMARKS, ci, per_probe_country, per_probe_default, set_arm,
)

FIGURES = PROJECT_ROOT / "outputs" / "figures" / "nocountry_agreement"
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"

# Slug -> the name used in iw_coordinates.csv.
IW_NAME = {"us": "united_states_of_america", "netherlands": "netherlands",
           "hungary": "hungary", "bosnia_and_herzegovina": "bosnia_and_herzegovina",
           "mexico": "mexico", "taiwan": "taiwan", "philippines": "philippines",
           "iran": "iran"}


# Eight plain, well-separated hues, one per probe country. The cluster palette
# could not be used as-is here: it gives EnglishSpeaking #4393c3 and Confucian
# #2166ac, two blues, so Taiwan read as a shade of the US. Ramps keyed on
# distance from the US were worse -- six of the eight countries sit between
# 0.111 and 0.146, so they came out as six near-identical greens.
PROBE_COLORS = {
    "us":                     "#4393c3",   # blue, kept from the cluster palette
    "netherlands":            "#ff7f0e",   # orange
    "hungary":                "#8c564b",   # brown
    "bosnia_and_herzegovina": "#9467bd",   # purple
    "mexico":                 "#e377c2",   # pink
    "taiwan":                 "#bcbd22",   # olive  (was a second blue)
    "philippines":            "#d62728",   # red
    "iran":                   "#2ca02c",   # green
}


def us_distances() -> dict[str, float]:
    """Euclidean distance from the US in Inglehart-Welzel space.

    The cluster palette colours by cluster identity, which is wrong for this
    chart: the question is how close a country is to the US, and two clusters
    can sit at the same distance. Taiwan (Confucian) is 0.125 from the US and
    the Netherlands (Protestant Europe) is 0.125 -- the cluster palette gave
    them two different blues, one of them nearly the US's own.

    Distance is measured from the US row itself, not from the EnglishSpeaking
    centroid that the csv's dist_from_english column holds, because the US is
    the probe country the whole analysis is about.
    """
    import csv
    import math
    rows = {r["normad_country"].strip().lower(): r
            for r in csv.DictReader(IW_COORDS.open())}
    us = rows[IW_NAME["us"]]
    ux, uy = float(us["sacsecval"]), float(us["resemaval"])
    out = {}
    for slug, name in IW_NAME.items():
        r = rows[name]
        out[slug] = math.hypot(float(r["sacsecval"]) - ux,
                               float(r["resemaval"]) - uy)
    return out


def distance_colors(cmap: str = "viridis", scale: str = "rank") -> dict[str, tuple]:
    """Shade by distance from the US: one end close, the other far.

    The variable encoded is a magnitude, so the ramp must be ordered -- a bar
    further along it has to mean further from the US and nothing else.
    Multi-hue ramps are fine for that as long as they are perceptually
    monotonic: viridis, plasma, magma and cividis all run dark-to-light while
    sweeping hue, so they look like a spectrum and still rank correctly. A
    literal rainbow (jet, hsv) does not -- its lightness rises and falls, so
    it invents visual groupings the distances do not have.

    'Blues' is still available via --cmap for the single-hue version.
    """
    d = us_distances()
    try:                                    # matplotlib >= 3.7
        ramp = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        ramp = cm.get_cmap(cmap)
    # Sequential single-hue ramps (Blues) run light->dark, so the US belongs at
    # the top; the perceptual ones run dark->light and the US belongs at the
    # bottom. Either way the US sits at the saturated end.
    reverse = cmap in {"Blues", "Greys", "Greens", "Oranges", "Purples", "Reds"}

    if scale == "value":
        lo, hi = min(d.values()), max(d.values())
        span = (hi - lo) or 1.0
        pos = {s: (v - lo) / span for s, v in d.items()}
    else:
        # By RANK, evenly spaced. The distances bunch -- six of the eight sit
        # between 0.111 and 0.146 -- so spacing by value paints those six
        # nearly the same colour and the chart reads as one hue. Rank spacing
        # keeps the ORDER (nearest to furthest) and spends the full ramp on
        # eight countries, at the cost that the colour gaps no longer show how
        # much further apart two countries are. Order is what the figure uses
        # colour for, so that trade is worth making; --scale value restores the
        # faithful-magnitude version.
        order = sorted(d, key=lambda s: d[s])
        pos = {s: i / (len(order) - 1) for i, s in enumerate(order)}

    if reverse:
        return {s: ramp(0.95 - 0.65 * p) for s, p in pos.items()}
    return {s: ramp(0.06 + 0.88 * p) for s, p in pos.items()}

# Fixed order, US first: it is the reference the whole analysis is about.
BAR_ORDER = [
    ("us",                     "EnglishSpeaking",  "US\n(English)"),
    ("netherlands",            "ProtestantEurope", "Netherlands\n(Protestant Eur.)"),
    ("hungary",                "CatholicEurope",   "Hungary\n(Catholic Eur.)"),
    ("bosnia_and_herzegovina", "Orthodox",         "Bosnia & Herz.\n(Orthodox)"),
    ("mexico",                 "LatinAmerica",     "Mexico\n(Latin America)"),
    ("taiwan",                 "Confucian",        "Taiwan\n(Confucian)"),
    ("philippines",            "SouthAsia",        "Philippines\n(South Asia)"),
    ("iran",                   "AfricanIslamic",   "Iran\n(African-Islamic)"),
]

# (condition, model) -> panel title
CELLS = [
    ("sftdpo_aya_nocult", "8b",     "Llama-3.1-8B, SFT+DPO (generic)"),
    ("sftdpo_aya_nocult", "gemma4", "Gemma-4-12B, SFT+DPO (generic)"),
    ("tulu3_dpo",         "8b",     "Tulu-3-8B, DPO"),
]

# --off-the-shelf plots the vendors' own instruct releases: no adapter of ours,
# so the arm is carried entirely by model_size. The condition slug differs by
# when the run was made -- "base" for the Llama/Gemma runs, "offshelf" for
# everything after the rename -- and path_for just concatenates it, so both
# spellings have to be listed per row rather than assumed.
#
# All four are scored with --raw-prompt (the chat template bypassed), so the
# panels are directly comparable to each other and to the pretrained runs.
OTS_CELLS = [
    ("base",       "8b_instruct",     "Llama-3.1-8B-Instruct"),
    ("tulu3_dpo",  "8b",              "Tulu-3-DPO-8B"),
    ("offshelf",   "olmo3_instruct",  "OLMo-3-7B-Instruct"),
    ("offshelf",   "qwen35_instruct", "Qwen3.5-9B"),
]

BENCH_TITLE = {"normad": "NormAd", "culturalbench": "CulturalBench",
               "blend": "BLEnD"}


def draw(ax, rates: dict, bench: str, title: str, show_ylabel: bool,
         baseline: str = "chance", sort: bool = True,
         color_by: str = "probe", cmap: str = "viridis",
         scale: str = "rank", metric: str = "blank",
         lo: float | None = None) -> None:
    xs, heights, errs, colors, labels = [], [], [], [], []
    PALETTE = distance_colors(cmap, scale) if color_by == "distance" else {}
    entries = [(slug, cluster, label) for slug, cluster, label in BAR_ORDER
               if slug in rates]
    if sort:
        # Sorted descending, so each panel reads greatest-to-least. The cost is
        # that a country sits at a different x position in each panel, so the
        # panels can no longer be compared by position -- read them one at a
        # time, or pass --no-sort for the fixed cluster order.
        entries.sort(key=lambda e: -rates[e[0]][0])
    for i, (slug, cluster, label) in enumerate(entries):
        v, n = rates[slug]
        xs.append(i)
        heights.append(v)
        errs.append(ci(v, n))
        colors.append(PROBE_COLORS[slug] if color_by == "probe"
                      else PALETTE[slug] if color_by == "distance"
                      else CLUSTER_COLORS[cluster])
        labels.append(label)

    bars = ax.bar(xs, heights, width=0.72, color=colors,
                  yerr=errs, capsize=3,
                  error_kw={"ecolor": "#444444", "elinewidth": 1.1})
    # Direct value labels: with eight bars the axis already names them, so the
    # number on each bar replaces reading heights off the grid.
    for b, v, e in zip(bars, heights, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#222222")

    # Chance agreement: NormAd and CulturalBench are yes/no, BLEnD is 4-way, so
    # the floor a bar has to clear is not the same across benchmarks and has to
    # be drawn rather than assumed.
    chance = 0.25 if bench == "blend" else 0.5
    if baseline == "chance":
        # The axis starts at chance, so bar LENGTH reads as "how far above
        # chance" rather than as the raw agreement rate. That is a truncated
        # bar axis, normally a distortion -- it is defensible only because
        # chance is a real floor for this measure, not an arbitrary crop, and
        # because every bar is still labelled with its actual value. The
        # dashed line is dropped since the axis itself now marks it.
        #
        # `lo` overrides the floor when some bar dips below chance. Clipping a
        # sub-chance bar to zero height would read as "exactly at chance",
        # which is a different claim, so the axis drops to fit it and the
        # chance line is drawn back in to keep the reference visible.
        floor = chance if lo is None else min(lo, chance)
        ax.set_ylim(floor, 1.0)
        if floor < chance:
            ax.axhline(chance, color="#888888", lw=1.0, ls=(0, (4, 3)),
                       zorder=0)
            ax.text(-0.62, chance + 0.012, f"chance {chance:.2f}",
                    ha="left", va="bottom", fontsize=7, color="#666666")
    else:
        ax.set_ylim(0, 1.0)
        ax.axhline(chance, color="#888888", lw=1.0, ls=(0, (4, 3)), zorder=0)
        # Left of the first bar and clear of the line: putting it at the right
        # edge ran it into the dashes and the axis frame.
        ax.text(-0.62, chance + 0.015, f"chance {chance:.2f}",
                ha="left", va="bottom", fontsize=7, color="#666666")

    ax.set_xlim(-0.7, len(labels) - 0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_title(title, fontsize=10)
    if show_ylabel:
        lab = ("Default rate: P(pred == swapped pred)" if metric == "default"
               else "Agreement with country-free prediction")
        if baseline == "chance" and (lo is None or lo >= chance):
            lab += f"\n(axis starts at chance, {chance:.2f})"
        ax.set_ylabel(lab, fontsize=9)
    # No shading caption: the ramp is decoration on top of an axis that already
    # carries the number and tick labels that already carry identity, so
    # nothing is lost by leaving it unexplained in the figure itself. State it
    # in the caption where the figure is used.
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="normad", choices=list(BENCHMARKS))
    ap.add_argument("--arm", default="raw", choices=["raw", "sys", "plain"],
                    help="Prompt-format arm to read, which is the last element "
                         "of every filename: raw for the --raw-prompt runs "
                         "(_raw), sys for the chat-templated ones (_sys), "
                         "plain for neither. Arms must not be mixed in one "
                         "figure -- agreement between a templated probe and a "
                         "raw baseline measures the prompt format.")
    ap.add_argument("--combined", action="store_true",
                    help="One file with the three panels side by side instead "
                         "of three separate files.")
    ap.add_argument("--errors-only", action="store_true",
                    help="Condition on the model's errors.")
    ap.add_argument("--group", default=None)
    ap.add_argument("--metric", default="blank", choices=["blank", "default"],
                    help="'blank' (default) is agreement between the swapped "
                         "prediction and the country-free one. 'default' is the "
                         "classic default rate P(pred == probe_pred), which "
                         "needs no country ablation but is unreliable on BLEnD "
                         "-- see per_probe_default().")
    ap.add_argument("--color-by", default="probe",
                    choices=["probe", "distance", "cluster"],
                    help="'probe' (default) gives each country its own plain "
                         "hue. 'distance' shades a ramp by Inglehart-Welzel "
                         "distance from the US. 'cluster' uses the categorical "
                         "cluster palette, in which Taiwan is a second blue.")
    ap.add_argument("--cmap", default="viridis",
                    help="Colormap for --color-by distance. Perceptually "
                         "ordered multi-hue options: viridis, plasma, magma, "
                         "cividis. Single-hue: Blues, Greens, Purples. Avoid "
                         "jet/hsv -- their lightness is not monotonic, so they "
                         "imply groupings the distances do not have.")
    ap.add_argument("--scale", default="rank", choices=["rank", "value"],
                    help="How distance maps onto the ramp. 'rank' (default) "
                         "spaces the eight countries evenly so the colours are "
                         "distinct; 'value' spaces them by actual distance, "
                         "which is faithful but paints the six mid-distance "
                         "countries nearly the same colour.")
    ap.add_argument("--no-sort", action="store_true",
                    help="Keep the fixed cluster order (US first) instead of "
                         "sorting each panel greatest-to-least. Fixed order is "
                         "the only way to compare panels by position.")
    ap.add_argument("--baseline", default="chance", choices=["chance", "zero"],
                    help="Where the y axis starts. 'chance' (default) makes bar "
                         "length read as distance above chance; 'zero' is the "
                         "untruncated axis.")
    ap.add_argument("--off-the-shelf", action="store_true",
                    help="Plot the vendors' own off-the-shelf instruct "
                         "releases (raw-prompted) instead of our "
                         "post-trained conditions.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="model_size labels to drop from the cell list, "
                         "space separated. Used to pull a model whose runs "
                         "are known bad without editing the cell tables.")
    args = ap.parse_args()
    # Must happen before any path_for() call, since it selects the filenames.
    set_arm(args.arm)

    bench = args.benchmark
    FIGURES.mkdir(parents=True, exist_ok=True)
    cells = OTS_CELLS if args.off_the_shelf else CELLS
    cells = [c for c in cells if c[1] not in args.exclude]
    suffix = ("_ots" if args.off_the_shelf else "") + \
             ("_defaultrate" if args.metric == "default" else "") + \
             ("_errors" if args.errors_only else "") + \
             ("_fromzero" if args.baseline == "zero" else "") + \
             ("_fixedorder" if args.no_sort else "") + \
             ("" if args.color_by == "probe" else f"_{args.color_by}color") + \
             ("" if args.cmap == "viridis" or args.color_by != "distance"
              else f"_{args.cmap.lower()}") + \
             ("_byvalue" if args.scale == "value"
                            and args.color_by == "distance" else "")

    found = []
    for cond, size, title in cells:
        fn = per_probe_default if args.metric == "default" else per_probe_country
        r = fn(bench, cond, size, args.group, args.errors_only)
        if r is None:
            print(f"  skipped {bench} {cond}/{size}: missing runs")
            continue
        found.append((cond, size, title, r))

    if not found:
        print(f"no {bench} data — run {BENCHMARKS[bench]['job']}")
        return

    # The error bar, not just the point, has to fit: a bar whose value clears
    # chance but whose lower whisker does not would still be clipped. Computed
    # across every panel because combined mode shares one y axis, and left at
    # None when nothing dips, so the usual chance-origin axis is unchanged.
    dips = [v - ci(v, n) for _, _, _, r in found for v, n in r.values()]
    lo = min(dips) - 0.02 if dips and min(dips) < (0.25 if bench == "blend"
                                                  else 0.5) else None

    if args.combined:
        fig, axes = plt.subplots(1, len(found), figsize=(8.0 * len(found), 4.8),
                                 sharey=True)
        if len(found) == 1:
            axes = [axes]
        for ax, (cond, size, title, r) in zip(axes, found):
            draw(ax, r, bench, title, show_ylabel=(ax is axes[0]),
                 baseline=args.baseline, sort=not args.no_sort,
                 color_by=args.color_by, cmap=args.cmap, scale=args.scale,
                 metric=args.metric, lo=lo)
        head = ("default rate P(pred == swapped pred) for each probe country"
                if args.metric == "default"
                else "agreement between the country-free prediction and each "
                     "probe country")
        fig.suptitle(f"{BENCH_TITLE[bench]}: {head}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = FIGURES / f"nocountry_agree_{bench}_combined{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")
        return

    for cond, size, title, r in found:
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        # Same floor as the combined figure rather than a per-panel one, so
        # the single-cell PNGs stay comparable to each other and to it.
        draw(ax, r, bench, f"{BENCH_TITLE[bench]} — {title}", show_ylabel=True,
             baseline=args.baseline, sort=not args.no_sort,
             color_by=args.color_by, cmap=args.cmap, scale=args.scale,
             metric=args.metric, lo=lo)
        fig.tight_layout()
        out = FIGURES / f"nocountry_agree_{bench}_{cond}_{size}{suffix}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
