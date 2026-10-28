#!/usr/bin/env python
"""Norm-only vs scenario-only on NormAd.

    APPLY   normad_{cond}_{size}_fs2_yn.json              country + scenario
    KNOW    normad_{cond}_{size}_rotknow[_probe].json     country + norm

Same 1,697 examples in the same order, crossed per example.

Pairing has no example IDs to rely on: both arms write rows in dataset order
after dropping neutral and holdout countries, so row i matches row i. align()
checks the (country, gold) sequences agree and refuses rather than reporting a
silently misaligned table.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"

PROBES = [("us", "US (English)"),
          ("netherlands", "Netherlands (Protestant Eur.)"),
          ("hungary", "Hungary (Catholic Eur.)"),
          ("bosnia_and_herzegovina", "Bosnia (Orthodox)"),
          ("mexico", "Mexico (Latin America)"),
          ("taiwan", "Taiwan (Confucian)"),
          ("philippines", "Philippines (South Asia)"),
          ("iran", "Iran (African-Islamic)")]

_US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united_states", "america",
               "united_states_of_america", "united states"}


def _canon(c: str) -> str:
    c = str(c).strip().lower().replace(" ", "_").replace("-", "_")
    return "us" if c in _US_ALIASES else c


def same_country(a: str, b: str) -> bool:
    return _canon(a) == _canon(b)


def know_path(cond: str, size: str, slug: str | None) -> Path:
    if slug is None:
        return BEHAVIORAL / f"normad_{cond}_{size}_rotknow.json"
    sfx = "_usprobe" if slug == "us" else f"_{slug}probe"
    return BEHAVIORAL / f"normad_{cond}_{size}_rotknow{sfx}.json"


def load_apply(cond: str, size: str):
    p = BEHAVIORAL / f"normad_{cond}_{size}_fs2_yn.json"
    return json.load(p.open())["predictions"] if p.exists() else None


def load_know(cond: str, size: str):
    """Any probe file serves: the own-country answers are identical in all of
    them, only the swapped country differs."""
    for slug in ["us"] + [s for s, _ in PROBES if s != "us"] + [None]:
        p = know_path(cond, size, slug)
        if p.exists():
            return json.load(p.open())
    return None


def align(a, b) -> None:
    if len(a) != len(b):
        raise SystemExit(f"APPLY has {len(a)} rows, KNOW has {len(b)} -- "
                         "cannot pair by position")
    bad = [i for i, (x, y) in enumerate(zip(a, b))
           if x["country"] != y["country"] or x["gold"] != y["gold"]]
    if bad:
        raise SystemExit("APPLY and KNOW disagree on (country, gold) at "
                         f"{len(bad)} rows, first at index {bad[0]} -- "
                         "cannot pair by position")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: under the null, b and c are a fair coin over
    b + c trials. Exact rather than chi-square because the cells can be small."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def margin(scores) -> float:
    """scores is [yes_logprob, no_logprob], so this is positive for yes."""
    return scores[0] - scores[1]


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, average ranks for ties. Used instead of Pearson because
    the yes-minus-no margins are not on a meaningful linear scale -- only their
    ordering over norms is."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def nocountry_vs_countries(cond: str, size: str) -> None:
    """Whose norms does the model hold by default?

    The country-free prompt asks the same question with no country named, so
    its answers are the model's unprompted normative defaults. Comparing those
    to each probe country's answers, over the same 1,697 norms, says which
    country's norms the defaults look like. If the US column sits above the
    other seven, the defaults are US-like; if all eight are level, the model is
    not conditioning on the country at all and nothing here is US-specific.

    Reported two ways: agreement of the yes/no decisions, and Spearman
    correlation of the yes-minus-no margins. The margins matter because the
    decisions can be near-constant -- every NormAd norm is a real norm, so
    "yes" is a defensible answer to all of them -- while the margins still rank
    the norms by how well each fits.
    """
    found = [(s, l) for s, l in PROBES if know_path(cond, size, s).exists()]
    if not found:
        return
    # nc_pred rides inside the probe files when one run did both; a
    # country-free-only run writes it to its own file instead.
    first = json.load(know_path(cond, size, found[0][0]).open())["records"]
    if first and "nc_pred" in first[0]:
        nc = None
    else:
        ncp = BEHAVIORAL / f"normad_{cond}_{size}_rotknow_nocountry.json"
        if not ncp.exists():
            return
        nc = json.load(ncp.open())["records"]

    print(f"\n  No country vs each country  (n paired against each)")
    print(f"    {'country':32s} {'agree':>7s} {'margin r':>9s} {'yes%':>7s} {'n':>6s}")
    nc_yes = sum(r["nc_pred"] == "yes" for r in nc) / len(nc)
    for slug, label in found:
        d = json.load(know_path(cond, size, slug).open())["records"]
        align(d, nc)
        rows = [(a, b) for a, b in zip(d, nc) if "us_pred" in a]
        if not rows:
            print(f"    {label:32s} {'--':>7s}")
            continue
        agree = sum(a["us_pred"] == b["nc_pred"] for a, b in rows) / len(rows)
        r = spearman([margin(a["us_scores"]) for a, _ in rows],
                     [margin(b["nc_scores"]) for _, b in rows])
        pyes = sum(a["us_pred"] == "yes" for a, _ in rows) / len(rows)
        print(f"    {label:32s} {agree * 100:6.1f}% {r:9.3f} {pyes * 100:6.1f}% "
              f"{len(rows):6d}")
    print(f"    no-country yes-rate = {nc_yes * 100:.1f}%")
    print("    (agree = same yes/no as the country-free prompt; margin r = Spearman\n"
          "     over the yes-minus-no margins, which survives a near-constant decision.)")


def report(cond: str, size: str) -> None:
    print(f"\n{'=' * 70}\n{cond}  /  {size}\n{'=' * 70}")
    apply_rows = load_apply(cond, size)
    if apply_rows is None:
        print(f"  APPLY missing: normad_{cond}_{size}_fs2_yn.json")
        return
    app_acc = sum(p["pred"] == p["gold"] for p in apply_rows) / len(apply_rows)
    print(f"\n  APPLY  scenario, accuracy = {app_acc * 100:5.1f}%   "
          f"n={len(apply_rows)}")

    know = load_know(cond, size)
    if know is None:
        print(f"  KNOW   missing: normad_{cond}_{size}_rotknow*.json")
        return
    print(f"  KNOW   norm, says yes     = {know['yes_rate'] * 100:5.1f}%   "
          f"n={know['n']}")

    recs = know["records"]
    align(apply_rows, recs)

    # yes on the norm x correct on the scenario
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for a, r in zip(apply_rows, recs):
        cells[(r["pred"] == "yes", a["pred"] == a["gold"])] += 1
    n = sum(cells.values())
    nk = cells[(True, True)] + cells[(True, False)]

    print(f"\n  {'':24s} {'scenario right':>15s} {'scenario wrong':>15s}")
    print(f"    {'norm: yes':22s} {cells[(True, True)]:15d} {cells[(True, False)]:15d}")
    print(f"    {'norm: no':22s} {cells[(False, True)]:15d} {cells[(False, False)]:15d}")

    p_k = cells[(True, True)] / nk if nk else float("nan")
    p_n = cells[(False, True)] / (n - nk) if n - nk else float("nan")
    print(f"\n    P(scenario right | norm yes) = {p_k:.3f}")
    print(f"    P(scenario right | norm no)  = {p_n:.3f}")
    print(f"    difference                   = {p_k - p_n:+.3f}")

    b, c = cells[(True, False)], cells[(False, True)]
    print(f"\n    norm yes / scenario wrong = {b}    norm no / scenario right = {c}")
    print(f"    McNemar exact two-sided p = {mcnemar_exact(b, c):.3g}")

    # country swap
    found = [(s, l) for s, l in PROBES if know_path(cond, size, s).exists()]
    if found:
        print(f"\n  Same norm, country swapped")
        print(f"    {'probe country':32s} {'yes%':>7s} {'agree':>7s} {'own yes%':>9s} {'n':>6s}")
        for slug, label in found:
            d = json.load(know_path(cond, size, slug).open())
            rows = [r for r in d["records"]
                    if "us_pred" in r and not same_country(r["country"], slug)]
            if not rows:
                print(f"    {label:32s} {'--':>7s}")
                continue
            print(f"    {label:32s} "
                  f"{sum(r['us_pred'] == 'yes' for r in rows) / len(rows) * 100:6.1f}% "
                  f"{sum(r['us_pred'] == r['pred'] for r in rows) / len(rows) * 100:6.1f}% "
                  f"{sum(r['pred'] == 'yes' for r in rows) / len(rows) * 100:8.1f}% "
                  f"{len(rows):6d}")

    nocountry_vs_countries(cond, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+",
                    default=["sftdpo_aya_cult", "sftdpo_aya_nocult", "tulu3_dpo"])
    ap.add_argument("--sizes", nargs="+", default=["8b", "gemma4"])
    args = ap.parse_args()
    for size in args.sizes:
        for cond in args.conditions:
            report(cond, size)
    print()


if __name__ == "__main__":
    main()
