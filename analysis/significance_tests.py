"""Bootstrap CIs + permutation tests for the NormAd accuracy gap-shifts in Table 2.

For each of the four comparisons in the paper:
    C1 → C2 (SFT alone)
    C1 → C3 (DPO alone)
    C1 → C4 (full pipeline)
    C2 → C4 (DPO on top of SFT)
we compute:
  - Observed gap-shift = Western Δ − Non-Western Δ
  - 95% bootstrap CI via country-level cluster bootstrap (default) or item-level
    paired bootstrap (--bootstrap item)
  - 2-sided permutation p-value via country-level relabeling: randomly assign each
    country to W or NW preserving the empirical count of each group, recompute
    gap-shift, repeat. p = fraction of permutations with |gap-shift_perm| >= |observed|

Rationale for country-level bootstrap as default: items within a NormAd country share
cultural context, scenario style, and prompt structure. Treating them as i.i.d. for
item-level bootstrap underestimates variance. Resampling at the country level
respects within-country correlation and gives a more conservative CI.

Reads:
  data/iw_coordinates.csv             ← per-country I-W cluster (Western if cluster
                                        ∈ {EnglishSpeaking, ProtestantEurope})
  outputs/behavioral/normad_*.json    ← per-condition predictions

Outputs:
  - Markdown table to stdout (pasteable into appendix or main text)

Usage:
    python3 analysis/significance_tests.py
    python3 analysis/significance_tests.py --n-boot 10000 --n-perm 10000
    python3 analysis/significance_tests.py --bootstrap item  # item-level bootstrap instead
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
IW_COORDS = PROJECT_ROOT / "data" / "iw_coordinates.csv"

# Same Western definition used by every other analysis script — must match the paper.
US_SIMILAR_CLUSTERS = {"EnglishSpeaking", "ProtestantEurope"}

# NormAd uses short forms for some countries (UK, US) where the CSV uses long
# forms (united_kingdom, united_states_of_america). Map short → CSV-canonical
# so these items don't get silently dropped.
NORMAD_ALIASES = {
    "UK": "united_kingdom",
    "US": "united_states_of_america",
    "USA": "united_states_of_america",
}


def _lookup_cluster(country: str, country_to_cluster: dict[str, str]) -> str | None:
    """Resolve a NormAd country to its I-W cluster with alias + case fallback."""
    if country in country_to_cluster:
        return country_to_cluster[country]
    lc = country.lower()
    if lc in country_to_cluster:
        return country_to_cluster[lc]
    alias = NORMAD_ALIASES.get(country) or NORMAD_ALIASES.get(lc)
    if alias and alias in country_to_cluster:
        return country_to_cluster[alias]
    return None

DEFAULT_COMPARISONS = [
    ("base",   "sft",     "C1 → C2 (SFT alone)"),
    ("base",   "dpo",     "C1 → C3 (DPO alone)"),
    ("base",   "sftdpo",  "C1 → C4 (full pipeline)"),
    ("sft",    "sftdpo",  "C2 → C4 (DPO on top of SFT)"),
]


def load_country_to_cluster(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. "
                 "Run analysis/culturemapping/compute_iw_coords.py first.")
    out: dict[str, str] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["normad_country"]:
                out[r["normad_country"]] = r["cluster"]
    return out


def load_condition(cond: str, country_to_cluster: dict[str, str],
                   size_suffix: str = "") -> list[tuple[int, str, str]]:
    """Return list of (correct, country, group_label) per prediction for one condition.

    `group_label` is 'W' or 'NW' derived from the country's I-W cluster (re-derived
    from data/iw_coordinates.csv rather than using the pre-computed `group` field
    in the JSON, which was generated under the old author-judgment partition).
    """
    path = BEHAVIORAL_DIR / f"normad_{cond}{size_suffix}.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found")
    preds = json.loads(path.read_text()).get("predictions", [])
    items: list[tuple[int, str, str]] = []
    for p in preds:
        country = p.get("country", "")
        cluster = _lookup_cluster(country, country_to_cluster)
        if cluster is None:
            # Skip rather than error — NormAd has a few countries not in WVS Wave 7.
            continue
        group = "W" if cluster in US_SIMILAR_CLUSTERS else "NW"
        correct = int(p["gold"] == p["pred"])
        items.append((correct, country, group))
    return items


def compute_deltas(before_items: list[tuple[int, str, str]],
                   after_items: list[tuple[int, str, str]]) -> tuple[float, float, float]:
    """Return (W_delta, NW_delta, gap_shift) for one (before, after) pair on the
    given paired item lists. Gap-shift = W_delta − NW_delta."""
    def acc_by_group(items):
        agg = defaultdict(lambda: [0, 0])  # group → [correct, total]
        for correct, _, group in items:
            agg[group][0] += correct
            agg[group][1] += 1
        return {g: (c / t) if t > 0 else float("nan")
                for g, (c, t) in agg.items()}
    a_b = acc_by_group(before_items)
    a_a = acc_by_group(after_items)
    w_delta = a_a.get("W", float("nan")) - a_b.get("W", float("nan"))
    nw_delta = a_a.get("NW", float("nan")) - a_b.get("NW", float("nan"))
    return w_delta, nw_delta, w_delta - nw_delta


def bootstrap_country(before_items, after_items, n_boot, rng):
    """Country-level cluster bootstrap. Resample countries with replacement, take
    all their items, compute gap-shift. Respects within-country correlation."""
    by_country_b: dict[str, list] = defaultdict(list)
    by_country_a: dict[str, list] = defaultdict(list)
    for b, a in zip(before_items, after_items):
        by_country_b[b[1]].append(b)
        by_country_a[a[1]].append(a)
    countries = sorted(by_country_b.keys())
    n_c = len(countries)

    gap_shifts = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(countries, size=n_c, replace=True)
        b_resample, a_resample = [], []
        for c in sampled:
            b_resample.extend(by_country_b[c])
            a_resample.extend(by_country_a[c])
        _, _, gs = compute_deltas(b_resample, a_resample)
        gap_shifts[i] = gs
    return gap_shifts


def bootstrap_item(before_items, after_items, n_boot, rng):
    """Item-level paired bootstrap. Resample item indices, compute gap-shift on
    the resampled (before, after) pair."""
    n = len(before_items)
    gap_shifts = np.empty(n_boot)
    for i in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        b_resample = [before_items[j] for j in idxs]
        a_resample = [after_items[j] for j in idxs]
        _, _, gs = compute_deltas(b_resample, a_resample)
        gap_shifts[i] = gs
    return gap_shifts


def permutation_country(before_items, after_items, n_perm, rng) -> tuple[np.ndarray, float]:
    """Country-level permutation null. Randomly relabel each country as W or NW
    while preserving the empirical count of each group, recompute gap-shift.

    Returns (perm_gap_shifts, two_sided_p_value)."""
    # Original country → group map
    country_group = {}
    for _, country, group in before_items:
        country_group[country] = group
    countries = sorted(country_group.keys())
    n_W = sum(1 for c in countries if country_group[c] == "W")

    observed_gs = compute_deltas(before_items, after_items)[2]

    perm_gs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(countries)
        perm_group = {c: ("W" if k < n_W else "NW") for k, c in enumerate(perm)}
        b_perm = [(c, ct, perm_group[ct]) for c, ct, _ in before_items]
        a_perm = [(c, ct, perm_group[ct]) for c, ct, _ in after_items]
        _, _, gs = compute_deltas(b_perm, a_perm)
        perm_gs[i] = gs

    p = float(np.mean(np.abs(perm_gs) >= np.abs(observed_gs)))
    return perm_gs, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", choices=["country", "item"], default="country",
                        help="country (default, more conservative) or item-level bootstrap")
    parser.add_argument("--model-size", choices=["3b", "8b"], default="3b")
    args = parser.parse_args()

    size_suffix = "_8b" if args.model_size == "8b" else ""
    rng = np.random.default_rng(args.seed)
    country_to_cluster = load_country_to_cluster(IW_COORDS)

    # Load every condition that appears in any comparison
    conditions_needed = {c for a, b, _ in DEFAULT_COMPARISONS for c in (a, b)}
    by_cond = {cond: load_condition(cond, country_to_cluster, size_suffix)
               for cond in conditions_needed}

    # Sanity: all conditions should have the same item count (paired bootstrap requirement)
    sizes = {cond: len(items) for cond, items in by_cond.items()}
    if len(set(sizes.values())) > 1:
        print(f"# WARN: item counts differ across conditions: {sizes}", file=sys.stderr)
        print(f"# Paired bootstrap requires equal-length, positionally-aligned items.",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n# Statistical tests for Table 2 gap-shifts")
    print(f"# Bootstrap: {args.bootstrap}-level, n={args.n_boot}; "
          f"Permutation: country-level, n={args.n_perm}; seed={args.seed}\n")

    boot_fn = bootstrap_country if args.bootstrap == "country" else bootstrap_item

    rows = []
    for before, after, label in DEFAULT_COMPARISONS:
        b_items = by_cond[before]
        a_items = by_cond[after]
        w_d, nw_d, gs = compute_deltas(b_items, a_items)
        boots = boot_fn(b_items, a_items, args.n_boot, rng)
        n_nan = int(np.isnan(boots).sum())
        if n_nan > 0:
            print(f"# NOTE: {n_nan}/{args.n_boot} {args.bootstrap}-bootstrap iterations "
                  f"for '{label}' produced NaN (a group had no items in that resample); "
                  f"using nanpercentile.", file=sys.stderr)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        perm_gs, p = permutation_country(b_items, a_items, args.n_perm, rng)
        ci_excludes_zero = (lo > 0) or (hi < 0)
        rows.append({
            "label": label, "w_d": w_d, "nw_d": nw_d, "gs": gs,
            "lo": lo, "hi": hi, "p": p, "sig": ci_excludes_zero,
        })

    # Markdown table for paste-into-paper
    print(f"| Comparison | Western Δ | Non-Western Δ | Gap-shift | 95% CI | perm p |")
    print(f"|---|---:|---:|---:|---|---:|")
    for r in rows:
        ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
        star = " *" if r["sig"] else ""
        print(f"| {r['label']} | {r['w_d']:+.3f} | {r['nw_d']:+.3f} | "
              f"{r['gs']:+.3f}{star} | {ci} | {r['p']:.4f} |")

    print(f"\n* = 95% CI for the gap-shift excludes 0 (i.e., observed asymmetry is")
    print(f"distinguishable from no-effect at α=0.05 under the chosen bootstrap).")
    print()
    print(f"Bootstrap CI obtained by resampling "
          f"{'countries (75 total)' if args.bootstrap == 'country' else 'items (2633 total)'} "
          f"with replacement, recomputing gap-shift on each resample. "
          f"Permutation null obtained by randomly relabeling each country as W or NW "
          f"while preserving the empirical count of each group, recomputing gap-shift, "
          f"and reporting the two-sided p as the fraction with |gap-shift_perm| ≥ |observed|.")


if __name__ == "__main__":
    main()
