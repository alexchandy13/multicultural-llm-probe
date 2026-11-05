"""BLEnD: whose answer does the model pick, with and without the country?

Every BLEnD question offers four options, each the correct answer for a
different country, so a pick names a culture. Two conditions:

  no country   the "in <country>" phrase is stripped, so the question has no
               right answer and the pick is pure default
  country, errors   the country is named; only the rows the model gets wrong
               can show a fallback, since the rest are the own-country answer

Both are plotted as a pick rate RELATIVE TO CHANCE, because the 16 countries
are not offered equally often and only one of four options can be picked:

    plotted = (times picked / times offered) / 0.25

Each question offers four options and takes one pick, so a country is picked
about a quarter of the times it appears; 0.25 is the chance rate, and dividing
by it puts 1.0 at "picked exactly as often as chance allows". The US in
sftdpo_aya_nocult/8b is offered 6,541 times and picked 3,441 -- 0.526 of the
time it is on the menu, or 2.10x chance. This is the same number as
share-of-picks / share-of-option-slots, which is how the tallies compute it,
but the per-question form is what the axis means.

The x axis is logarithmic. Ratios are multiplicative, so on a linear axis 2.0
sits one unit above 1.0 while its opposite 0.5 sits half a unit below, which
draws over-selection as the larger effect purely by arithmetic. On a log axis
they are equidistant from 1.0, which is what "twice as often" and "half as
often" deserve.

Colour marks one entity rather than encoding identity: sixteen categorical
hues would be a rainbow, and the chart is a magnitude ranking with the US as
the entity of interest. US takes the cluster palette's blue, everything else
is neutral ink.

    python analysis/blend_default_dumbbell.py
    python analysis/blend_default_dumbbell.py --include-dummy
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
from analysis.blend_error_defaults import tally
from analysis.blend_nocountry_defaults import defaults

FIGURES = PROJECT_ROOT / "outputs" / "figures" / "blend_defaults"

# The three cells the other figures use: both generic SFT+DPO arms and Tulu-3.
# The cultural arms are a separate comparison and only crowd this one, so they
# are behind --with-cult rather than in the default set.
CELLS = [
    ("sftdpo_aya_nocult", "8b",     "Llama-3.1-8B, SFT+DPO (generic)"),
    ("sftdpo_aya_nocult", "gemma4", "Gemma-4-12B, SFT+DPO (generic)"),
    ("tulu3_dpo",         "8b",     "Tulu-3-8B, DPO"),
]
CULT_CELLS = [
    ("sftdpo_aya_cult",   "8b",     "Llama-3.1-8B, SFT+DPO (cultural)"),
    ("sftdpo_aya_cult",   "gemma4", "Gemma-4-12B, SFT+DPO (cultural)"),
]

# --off-the-shelf swaps the whole set for the vendors' own instruct releases,
# all raw-prompted. The condition slug differs by when the run was made --
# "base" for the Llama/Gemma runs, "offshelf" after the rename -- so it is
# listed per row rather than assumed.
OTS_CELLS = [
    ("base",     "8b_instruct",     "Llama-3.1-8B-Instruct"),
    ("base",     "gemma4_instruct", "Gemma-4-12B-it"),
    ("offshelf", "olmo3_instruct",  "OLMo-3-7B-Instruct"),
    ("offshelf", "qwen35_instruct", "Qwen3.5-9B"),
]

US_COLOR = CLUSTER_COLORS["EnglishSpeaking"]
NC_COLOR = "#333333"      # no country: the default being measured
WC_COLOR = "#9a9a9a"      # country given: the control
PRETTY = {"US": "US", "UK": "UK", "South_Korea": "South Korea",
          "North_Korea": "North Korea", "West_Java": "West Java",
          "Northern_Nigeria": "Northern Nigeria"}


def ratios(cond: str, size: str, include_dummy: bool, scope: str):
    """{country: (no_country_ratio, country_given_ratio | None)}.

    scope="errors" counts only the rows the model got wrong. scope="all"
    counts every row, which mixes two different things into one number: a
    country's own questions are the ones it is the gold answer to, so being
    right on them pushes its ratio up. The US ratio under scope="all" is
    therefore part default and part accuracy on US questions, and is not
    comparable to the no-country ratio in the way the errors-only one is.
    """
    nc = defaults(cond, size)
    if nc is None:
        return None
    picked, avail, n = nc
    out = {}
    for c, k in picked.items():
        if c == "dummy" and not include_dummy:
            continue
        base = avail[c] / (4 * n)
        if base:
            out[c] = [(k / n) / base, None]

    wc = tally(cond, size, scope)
    if wc is not None:
        p2, a2, n2, slots, _acc, _off, _tk = wc
        for c, k in p2.items():
            if c in out and a2[c]:
                out[c][1] = (k / n2) / (a2[c] / slots)
    return out


def draw(ax, data: dict, title: str, as_rate: bool = False,
         vertical: bool = False) -> None:
    # Horizontal reads bottom-up (biggest at the top); vertical reads
    # left-to-right (biggest first), which is how a bar chart is scanned.
    order = sorted(data, key=lambda c: data[c][0], reverse=vertical)
    idx = range(len(order))
    # The ratio is (picked/offered)/0.25, so scaling by 0.25 recovers the raw
    # pick rate -- the share of the times a country is on the menu that it wins.
    k = 0.25 if as_rate else 1.0

    for i, c in zip(idx, order):
        a, b = data[c]
        hi = US_COLOR if c == "US" else NC_COLOR
        pos, far = ([i, i], [b * k, a * k]) if vertical else ([b * k, a * k], [i, i])
        if b is not None:
            ax.plot(*((pos, far) if vertical else (pos, far)),
                    color="#cccccc", lw=1.6, zorder=1, solid_capstyle="round")
            pt = ([i], [b * k]) if vertical else ([b * k], [i])
            ax.plot(*pt, "o", ms=5.5, color=WC_COLOR, zorder=2,
                    markeredgecolor="white", markeredgewidth=0.8)
        pt = ([i], [a * k]) if vertical else ([a * k], [i])
        ax.plot(*pt, "o", ms=7.5, color=hi, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)

    names = [PRETTY.get(c, c.replace("_", " ")) for c in order]
    val_lo, val_hi = 0.18 * k, 3.0 * k
    ticks = [t * k for t in (0.25, 0.5, 1.0, 2.0)]
    tick_labels = ([f"{t * k:.3f}".rstrip("0") for t in (0.25, 0.5, 1.0, 2.0)]
                   if as_rate else ["0.25×", "0.5×", "1×", "2×"])
    val_label = ("Pick rate: picked ÷ offered   (chance = 0.25)" if as_rate
                 else "Pick rate ÷ chance   (chance = 1 option in 4)")

    if vertical:
        ax.axhline(1.0 * k, color="#888888", lw=1.0, ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(names, fontsize=8.5, rotation=45, ha="right")
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_yscale("log")
        ax.set_ylim(val_lo, val_hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels, fontsize=8.5)
        ax.set_ylabel(val_label, fontsize=9)
        ax.grid(axis="y", color="#e8e8e8", lw=0.6)
        label_axis = ax.get_xticklabels()
        hide = ("top", "right")
    else:
        ax.axvline(1.0 * k, color="#888888", lw=1.0, ls=(0, (4, 3)), zorder=0)
        ax.set_yticks(list(idx))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(val_lo, val_hi)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8.5)
        ax.set_xlabel(val_label, fontsize=9)
        ax.grid(axis="x", color="#e8e8e8", lw=0.6)
        label_axis = ax.get_yticklabels()
        hide = ("top", "right", "left")

    for t, c in zip(label_axis, order):
        if c == "US":
            t.set_color(US_COLOR)
            t.set_fontweight("bold")

    ax.set_title(title, fontsize=10)
    ax.set_axisbelow(True)
    for side in hide:
        ax.spines[side].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-dummy", action="store_true",
                    help="Keep the fabricated 'dummy' option. It is a sanity "
                         "check -- the model avoids it, which shows it reads "
                         "the options rather than picking positionally -- not "
                         "a country, so it is dropped by default.")
    ap.add_argument("--with-cult", action="store_true",
                    help="Also include the two cultural SFT+DPO arms.")
    ap.add_argument("--off-the-shelf", action="store_true",
                    help="Plot the vendors' own off-the-shelf instruct "
                         "releases (raw-prompted) instead of our "
                         "post-trained conditions.")
    ap.add_argument("--combined", action="store_true",
                    help="All available cells in one file.")
    ap.add_argument("--vertical", action="store_true",
                    help="Countries on the x axis, value on the y axis.")
    ap.add_argument("--as-rate", action="store_true",
                    help="Plot the raw pick-rate (picked / offered, chance "
                         "0.25) instead of its ratio to chance.")
    ap.add_argument("--scope", default="errors", choices=["errors", "all"],
                    help="Which rows the country-named dot covers. 'errors' "
                         "(default) is the only place a fallback can show once "
                         "the country is given. 'all' includes the correct "
                         "answers, so a country's ratio also rises with "
                         "accuracy on its own questions.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="model_size labels to drop from the cell list, "
                         "space separated. Used to pull a model whose runs "
                         "are known bad without editing the cell tables.")
    args = ap.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)

    cells = (OTS_CELLS if args.off_the_shelf
             else CELLS + (CULT_CELLS if args.with_cult else []))
    cells = [c for c in cells if c[1] not in args.exclude]
    found = []
    for cond, size, title in cells:
        d = ratios(cond, size, args.include_dummy, args.scope)
        if d is None:
            print(f"  skipped {cond}/{size}: no country-ablation run")
            continue
        found.append((cond, size, title, d))
    if not found:
        print("no BLEnD country-ablation runs — "
              "run slurm/eval_blend_nocountry_job.sh")
        return

    legend = [
        plt.Line2D([], [], marker="o", ls="", ms=7.5, color=NC_COLOR,
                   label="No country named"),
        plt.Line2D([], [], marker="o", ls="", ms=5.5, color=WC_COLOR,
                   # "all rows" is the default reading of "country named", so
                   # it needs no qualifier; "errors only" does, since it drops
                   # the rows the model got right.
                   label=("Country named" if args.scope == "all"
                          else "Country named (errors only)")),
    ]
    sfx = ("_ots" if args.off_the_shelf else "") + \
          ("_withcult" if args.with_cult else "") + \
          ("_withdummy" if args.include_dummy else "") + \
          ("_allrows" if args.scope == "all" else "") + \
          ("_rate" if args.as_rate else "") + \
          ("_vertical" if args.vertical else "")

    if args.combined:
        n = len(found)
        # A vertical panel needs 16 rotated country labels along its x axis, so
        # it cannot be squeezed: five of them in one row came out 30 inches
        # wide and 5 tall, with every panel crushed. Wrap to a grid instead.
        # Horizontal panels share one x axis and tile fine in a single row.
        if args.vertical:
            ncol = min(3, n)
            nrow = -(-n // ncol)
            figsize = (7.2 * ncol, 5.6 * nrow)
        else:
            ncol, nrow = n, 1
            figsize = (4.3 * n, 5.4)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False,
                                 sharex=not args.vertical,
                                 sharey=args.vertical)
        flat = [a for row in axes for a in row]
        for ax, (_c, _s, title, d) in zip(flat, found):
            draw(ax, d, title, args.as_rate, args.vertical)
        for ax in flat[n:]:              # unused cells in a ragged grid
            ax.set_visible(False)
        if args.vertical:
            # sharey keeps the scale but still draws a y label on every panel.
            for row in axes:
                for ax in row[1:]:
                    ax.set_ylabel("")
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
                   fontsize=9)
        fig.suptitle("BLEnD: whose answer the model picks, relative to how "
                     "often that country is offered", fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = FIGURES / f"blend_defaults_dumbbell_combined{sfx}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")
        return

    for cond, size, title, d in found:
        fig, ax = plt.subplots(figsize=((8.6, 5.0) if args.vertical else (6.4, 5.4)))
        draw(ax, d, f"BLEnD — {title}", args.as_rate, args.vertical)
        # Upper left: the high-ratio rows sit top-RIGHT and the low-ratio rows
        # bottom-left, so this is the one empty corner. lower right lands on
        # the Ethiopia / Northern Nigeria markers.
        ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8.5)
        fig.tight_layout()
        out = FIGURES / f"blend_defaults_dumbbell_{cond}_{size}{sfx}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
