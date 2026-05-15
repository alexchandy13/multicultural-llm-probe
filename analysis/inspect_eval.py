"""Quick post-hoc inspection of a NormAd eval JSON.

Prints:
  - group counts (Western / Non-Western / Other)
  - per-group accuracy
  - per-country accuracy within Western and Non-Western
  - top/bottom "Other" countries by accuracy (the bulk of the dataset)
  - W vs. NW gap

Usage:
    python3.12 analysis/inspect_eval.py                  # all conditions found
    python3.12 analysis/inspect_eval.py sft              # one condition
    python3.12 analysis/inspect_eval.py base sft         # side-by-side
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"
CONDITION_ORDER = ["base", "sft", "dpo", "instruct"]

# Import the live W/NW sets + NormAd country map so re-grouping always
# tracks the current methodology, even on JSONs saved before the expansion.
sys.path.insert(0, str(PROJECT_ROOT))
from evaluate._common import WESTERN, NON_WESTERN
from evaluate.eval_normad import NORMAD_COUNTRY_MAP


def regroup(raw_country: str) -> tuple[str, str]:
    """(canonical_country, group) — re-derives both from a saved prediction's `country`.

    Handles both canonical-style values (e.g. "US") and raw NormAd lowercase
    values (e.g. "france") that historical JSONs may contain.
    """
    canon = NORMAD_COUNTRY_MAP.get(raw_country.lower(), raw_country)
    if canon in WESTERN:
        return canon, "Western"
    if canon in NON_WESTERN:
        return canon, "Non-Western"
    return canon, "Other"


def load(cond: str) -> dict | None:
    path = BEHAVIORAL_DIR / f"normad_{cond}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def per_country(predictions: list[dict], group: str) -> dict[str, tuple[int, int]]:
    """Return {country: (correct, total)} for predictions in the given (re-derived) group."""
    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in predictions:
        canon, g = regroup(p["country"])
        if g != group:
            continue
        rows[canon][0] += int(p["gold"] == p["pred"])
        rows[canon][1] += 1
    return {c: (r[0], r[1]) for c, r in rows.items()}


def fmt_pct(x):
    return "  —  " if x is None else f"{x:.3f}"


def print_condition(cond: str, data: dict):
    preds = data["predictions"]
    # Always re-group from raw country using the live map, so historical JSONs
    # automatically benefit when WESTERN/NON_WESTERN expands.
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in preds:
        _, g = regroup(p["country"])
        groups[g][0] += int(p["gold"] == p["pred"])
        groups[g][1] += 1

    total_n = sum(g[1] for g in groups.values())
    print(f"\n========== condition = {cond}  (n={total_n}, overall acc = {data['accuracy_overall']:.3f}) ==========")

    print("\n  Group totals (re-grouped with current map):")
    for g, (c, n) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"    {g:12s} {n:5d}  ({100*n/total_n:5.1f}%)   acc = {c/n:.3f}")

    w_correct, w_n = groups.get("Western", [0, 0])
    nw_correct, nw_n = groups.get("Non-Western", [0, 0])
    w_acc = (w_correct / w_n) if w_n else None
    nw_acc = (nw_correct / nw_n) if nw_n else None
    gap = (w_acc - nw_acc) if (w_acc is not None and nw_acc is not None) else None
    print(f"\n  W − NW gap = {fmt_pct(gap)}")

    for g in ("Western", "Non-Western"):
        rows = per_country(preds, g)
        if not rows:
            continue
        print(f"\n  Per-country accuracy [{g}]:")
        for country, (correct, n) in sorted(rows.items(), key=lambda x: -x[1][1]):
            print(f"    {country:20s} {correct/n:.3f}  ({correct}/{n})")

    # Top/bottom 5 "Other" countries by accuracy (only those with >= 20 examples
    # so single-instance outliers don't dominate).
    other = per_country(preds, "Other")
    other = {c: r for c, r in other.items() if r[1] >= 20}
    if other:
        ranked = sorted(other.items(), key=lambda x: -x[1][0] / x[1][1])
        print("\n  'Other' top 5 (acc, n>=20):")
        for country, (correct, n) in ranked[:5]:
            print(f"    {country:30s} {correct/n:.3f}  ({correct}/{n})")
        print("  'Other' bottom 5 (acc, n>=20):")
        for country, (correct, n) in ranked[-5:]:
            print(f"    {country:30s} {correct/n:.3f}  ({correct}/{n})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="*",
                        help="One or more of base / sft / dpo / instruct. "
                             "Default: every JSON found under outputs/behavioral/.")
    args = parser.parse_args()

    if args.conditions:
        wanted = args.conditions
    else:
        wanted = [c for c in CONDITION_ORDER if (BEHAVIORAL_DIR / f"normad_{c}.json").exists()]

    if not wanted:
        print("No NormAd eval JSON files found in", BEHAVIORAL_DIR)
        return

    for cond in wanted:
        data = load(cond)
        if data is None:
            print(f"\n[skip] {cond}: outputs/behavioral/normad_{cond}.json not found")
            continue
        print_condition(cond, data)


if __name__ == "__main__":
    main()
