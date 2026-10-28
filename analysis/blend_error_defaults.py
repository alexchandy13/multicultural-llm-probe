"""Which culture does the model fall back to when the country IS named?

analysis/blend_nocountry_defaults.py answers this for the country-free prompt,
where the question has no right answer and the pick simply names a culture.
With the country named there IS a right answer, so the question only applies
to the rows the model gets wrong: when it misses, whose answer does it reach
for instead?

The country-included runs do not record choice_countries -- only
country/gold/group/mcqid/pred/us_pred -- so the letter cannot be mapped to a
country from that file alone. The mapping is taken from the country-free run
of the same condition, which is sound only because the option order is
identical between the two: mcqid sequence, country, and gold letter all match
on every row (23,032/23,032), and choice_countries[gold] == own country on
every row. verify_join() re-checks all of that and refuses otherwise, since a
silent mismatch would relabel every pick.

BASE RATE. On an error row the gold option was available but, by definition,
not taken, so the model was choosing among the other three. Availability is
therefore counted over non-gold options only -- 3 slots per row, not 4.
Dividing by the 4-option rate would understate every ratio and, worse, would
penalise exactly the countries that appear most often as the gold answer.

    python analysis/blend_error_defaults.py
    python analysis/blend_error_defaults.py --scope all
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"

# tulu3_dpo is 8b-only (no Gemma release), so its gemma4 cell is
# reported as missing rather than silently dropped.
CONDS = ["sftdpo_aya_cult", "sftdpo_aya_nocult", "tulu3_dpo"]
MODELS = ["8b", "gemma4"]


def load(name: str):
    p = BEHAVIORAL / name
    return json.loads(p.read_text())["predictions"] if p.exists() else None


def verify_join(withc: list[dict], nc: list[dict], tag: str) -> None:
    """The option->country map comes from the other file, so prove they align."""
    if len(withc) != len(nc):
        raise SystemExit(f"{tag}: {len(withc)} vs {len(nc)} rows, cannot join")
    for field in ("mcqid", "country", "gold"):
        bad = [i for i, (a, b) in enumerate(zip(withc, nc)) if a[field] != b[field]]
        if bad:
            raise SystemExit(f"{tag}: {field} differs at {len(bad)} rows "
                             f"(first index {bad[0]}) -- option order is not "
                             "shared, so choice_countries cannot be reused")
    bad = [i for i, b in enumerate(nc)
           if str((b.get("choice_countries") or {}).get(b["gold"], "")).lower()
           != str(b["country"]).lower()]
    if bad:
        raise SystemExit(f"{tag}: choice_countries[gold] != country at "
                         f"{len(bad)} rows (first {bad[0]})")


def tally(cond: str, size: str, scope: str):
    """(picked, available, n_rows, accuracy) with the country in the prompt."""
    withc = load(f"blend_{cond}_{size}_nfs_usprobe.json")
    nc = load(f"blend_{cond}_{size}_nfs_nocountry.json")
    if withc is None or nc is None:
        return None
    verify_join(withc, nc, f"{cond}/{size}")

    picked, avail = Counter(), Counter()
    # offered/taken give the per-row conditional P(pick X | X is on the menu),
    # which is what the ratio is meant to express. The ratio divides two
    # aggregate shares and so assumes availability is unrelated to everything
    # else; the conditional assumes nothing. They match to within 0.01 on every
    # cell, but both are printed so that is visible rather than asserted.
    offered, taken = Counter(), Counter()
    n = n_correct = 0
    slots = 0
    for a, b in zip(withc, nc):
        cc = b.get("choice_countries") or {}
        if not cc:
            continue
        correct = a["pred"] == a["gold"]
        n_correct += correct
        n += 1
        if scope == "errors" and correct:
            continue
        who = cc.get(a["pred"])
        if who is None:
            continue
        picked[who] += 1
        # On an error the gold option was passed over, so the choosable set is
        # the three non-gold options. With scope=all every option counts.
        choosable = {L: c for L, c in cc.items()
                     if not (scope == "errors" and L == a["gold"])}
        for c in choosable.values():
            avail[c] += 1
            slots += 1
        for c in set(choosable.values()):
            offered[c] += 1
            taken[c] += (who == c)
    if not picked:
        return None
    return (picked, avail, sum(picked.values()), slots,
            (n_correct / n if n else 0.0), offered, taken)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="errors", choices=["errors", "all"],
                    help="'errors' (default) counts only rows the model got "
                         "wrong, which is the only place a fallback is visible "
                         "once the country is given. 'all' counts every row, so "
                         "the own-country answer dominates in proportion to "
                         "accuracy.")
    args = ap.parse_args()
    missing = []

    for size in MODELS:
        for cond in CONDS:
            t = tally(cond, size, args.scope)
            if t is None:
                missing.append(f"{cond}/{size}")
                continue
            picked, avail, n, slots, acc, offered, taken = t
            chance = 1 / 3 if args.scope == "errors" else 1 / 4
            print(f"\n### BLEnD {size} {cond}   country IN prompt, "
                  f"scope={args.scope}   accuracy {acc:.1%}   n={n}")
            print(f"{'country':18s} {'picked':>8s} {'share':>8s} "
                  f"{'base rate':>10s} {'ratio':>7s} {'P(pick|offered)':>16s} "
                  f"{'n offered':>10s}")
            for c, k in picked.most_common():
                share = k / n
                base = avail[c] / slots if slots else 0.0
                if not base:
                    continue
                cnd = taken[c] / offered[c] if offered[c] else float("nan")
                print(f"{c:18s} {k:8d} {share:7.1%} {base:10.1%} "
                      f"{share / base:7.2f} {cnd:16.3f} {offered[c]:10d}")
            print(f"{'':18s} {'':8s} {'':8s} {'':10s} {'':7s} "
                  f"{'chance ' + format(chance, '.3f'):>16s}")
    if missing:
        print(f"\nmissing runs: {', '.join(missing)}")


if __name__ == "__main__":
    main()
