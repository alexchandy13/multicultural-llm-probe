"""Which culture does the model answer with when no country is named?

Reads the BLEnD country-ablation runs. Because each option is the correct
answer for a different country, a country-free question has no right answer --
the model's pick names a culture. Two readings:

  defaults   the distribution over picked countries, against the base rate at
             which each country appears among the four options. A country
             picked well above its availability is a default; at its base rate
             it is being chosen no more than chance allows.
  agreement  P(nc_pred == us_pred), the same two-column comparison the NormAd
             ablation gives, for continuity with that analysis.

Needs slurm/eval_blend_nocountry_job.sh output:
  blend_{cond}_{size}_nfs_nocountry.json
alongside the existing blend_{cond}_{size}_nfs_usprobe.json.

    python analysis/blend_nocountry_defaults.py
    python analysis/blend_nocountry_defaults.py --mode agreement
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from math import sqrt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"

CONDS = ["sftdpo_aya_cult", "sftdpo_aya_nocult"]
MODELS = ["8b", "gemma4"]


def load(name: str):
    p = BEHAVIORAL / name
    return json.loads(p.read_text())["predictions"] if p.exists() else None


def ci(p: float, n: int) -> float:
    return 1.96 * sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def defaults(cond: str, size: str):
    """(picked Counter, available Counter, n) over country-free questions."""
    nc = load(f"blend_{cond}_{size}_nfs_nocountry.json")
    if nc is None:
        return None
    picked, avail, n = Counter(), Counter(), 0
    for x in nc:
        cc = x.get("choice_countries") or {}
        if not cc or x.get("pred_country") is None:
            continue
        picked[x["pred_country"]] += 1
        # each option is one country's answer, so every row offers four
        for c in cc.values():
            avail[c] += 1
        n += 1
    return (picked, avail, n) if n else None


def agreement(cond: str, size: str):
    us = load(f"blend_{cond}_{size}_nfs_usprobe.json")
    nc = load(f"blend_{cond}_{size}_nfs_nocountry.json")
    if us is None or nc is None:
        return None
    if len(us) != len(nc):
        raise ValueError(f"{cond}/{size}: {len(us)} vs {len(nc)} rows, "
                         f"cannot pair by index")
    pairs = [(u, c) for u, c in zip(us, nc)
             if u.get("us_pred") is not None and u.get("country") != "US"]
    n = len(pairs)
    if not n:
        return None
    return {
        "real_vs_us":    sum(u["pred"] == u["us_pred"] for u, _ in pairs) / n,
        "real_vs_blank": sum(u["pred"] == c["pred"] for u, c in pairs) / n,
        "us_vs_blank":   sum(u["us_pred"] == c["pred"] for u, c in pairs) / n,
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="defaults",
                    choices=["defaults", "agreement"])
    args = ap.parse_args()
    missing = []

    for size in MODELS:
        for cond in CONDS:
            if args.mode == "agreement":
                r = agreement(cond, size)
                if r is None:
                    missing.append(f"{cond}/{size}")
                    continue
                print(f"\n### BLEnD {size} {cond}   n={r['n']}")
                for k in ("real_vs_us", "real_vs_blank", "us_vs_blank"):
                    print(f"  {k:16s} {r[k]:.3f} ± {ci(r[k], r['n']):.3f}")
                continue

            d = defaults(cond, size)
            if d is None:
                missing.append(f"{cond}/{size}")
                continue
            picked, avail, n = d
            print(f"\n### BLEnD {size} {cond}   {n} country-free questions")
            print(f"{'country':18s} {'picked':>8s} {'share':>8s} "
                  f"{'base rate':>10s} {'ratio':>7s}")
            for c, k in picked.most_common():
                share = k / n
                base = avail[c] / (4 * n)      # 4 options offered per question
                print(f"{c:18s} {k:8d} {share:7.1%} {base:10.1%} "
                      f"{share / base:7.2f}" if base else "")
    if missing:
        print(f"\nmissing country-ablation runs: {', '.join(missing)}")
        print("run slurm/eval_blend_nocountry_job.sh (+ the gemma4 variant)")


if __name__ == "__main__":
    main()
