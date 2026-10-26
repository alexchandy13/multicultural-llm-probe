"""Does swapping in the US differ from giving no country at all?

For each condition, three agreement rates over the same examples:

  real vs US        P(pred == us_pred)         -- the usual US-default rate
  real vs blank     P(pred == nc_pred)         -- how much the country matters
  US vs blank       P(us_pred == nc_pred)      -- the one that settles it

If "US vs blank" is near 1.0, then telling the model it is the US and telling
it nothing produce the same answers, and a high US-default rate is evidence
that the model ignores the country -- not that it is anchored on the US. A low
value means the US token does something specific.

Reads the _fs2_yn setting. Needs the country-ablation runs from
slurm/eval_normad_nocountry_job.sh:
  normad_{cond}_{size}_fs2_yn_nocountry.json      (coherent mode)
  normad_{cond}_{size}_fs2_yn_nocountrydrop.json  (drop mode)
alongside the existing
  normad_{cond}_{size}_fs2_yn_usprobe.json

Both runs exclude the same holdout countries, so the 1,697 rows pair by index;
the loader checks that and refuses if they differ.

    python analysis/nocountry_agreement.py
    python analysis/nocountry_agreement.py --mode drop --group Non-Western
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"

CONDS = ["base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult",
         "sftdpo_aya_nocult", "tulu3_sft", "tulu3_dpo"]
MODELS = ["8b", "gemma4"]


def load(name: str):
    p = BEHAVIORAL / name
    return json.loads(p.read_text())["predictions"] if p.exists() else None


def ci(p: float, n: int) -> float:
    return 1.96 * sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def rates(cond: str, size: str, group: str | None, errors_only: bool,
          mode: str = "coherent"):
    sfx = "_nocountry" if mode == "coherent" else "_nocountrydrop"
    us = load(f"normad_{cond}_{size}_fs2_yn_usprobe.json")
    nc = load(f"normad_{cond}_{size}_fs2_yn{sfx}.json")
    if us is None or nc is None:
        return None
    if len(us) != len(nc):
        raise ValueError(f"{cond}/{size}: {len(us)} usprobe rows vs {len(nc)} "
                         f"nocountry rows -- different filters, cannot pair")
    pairs = [(u, n) for u, n in zip(us, nc)
             if u.get("us_pred") is not None and u.get("country") != "US"
             and (group is None or u.get("group") == group)]
    if errors_only:
        pairs = [(u, n) for u, n in pairs if u["pred"] != u["gold"]]
    if not pairs:
        return None
    n = len(pairs)
    return {
        "real_vs_us":    sum(u["pred"] == u["us_pred"] for u, _ in pairs) / n,
        "real_vs_blank": sum(u["pred"] == c["pred"] for u, c in pairs) / n,
        "us_vs_blank":   sum(u["us_pred"] == c["pred"] for u, c in pairs) / n,
        "n": n,
    }


REPS = {"us": "US (English)", "netherlands": "Netherlands (Protestant)",
        "hungary": "Hungary (Catholic)",
        "bosnia_and_herzegovina": "Bosnia (Orthodox)",
        "mexico": "Mexico (Latin America)", "taiwan": "Taiwan (Confucian)",
        "philippines": "Philippines (South Asia)",
        "iran": "Iran (African-Islamic)"}


def per_probe_country(cond: str, size: str, mode: str, group: str | None,
                      errors_only: bool):
    """{probe country: (P(probe_pred == blank_pred), n)}.

    The point of comparing all eight: if the blank-country prediction agrees
    with every probe country at about the same rate, the model is not anchored
    on any of them -- it is ignoring the country field. A country that agrees
    noticeably more is a genuine default.

    Each probe file stores its swapped prediction in us_pred regardless of
    which country was swapped in, and leaves it None for rows already from
    that country, so those rows drop out per country and n varies slightly.
    """
    sfx = "_nocountry" if mode == "coherent" else "_nocountrydrop"
    nc = load(f"normad_{cond}_{size}_fs2_yn{sfx}.json")
    if nc is None:
        return None
    out = {}
    for slug in REPS:
        pb = load(f"normad_{cond}_{size}_fs2_yn_{slug}probe.json")
        if pb is None:
            continue
        if len(pb) != len(nc):
            raise ValueError(f"{cond}/{size}/{slug}: {len(pb)} probe rows vs "
                             f"{len(nc)} nocountry rows, cannot pair")
        pairs = [(a, b) for a, b in zip(pb, nc)
                 if a.get("us_pred") is not None
                 and (group is None or a.get("group") == group)]
        if errors_only:
            pairs = [(a, b) for a, b in pairs if a["pred"] != a["gold"]]
        if pairs:
            agree = sum(a["us_pred"] == b["pred"] for a, b in pairs)
            out[slug] = (agree / len(pairs), len(pairs))
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None,
                    help="Restrict to one group field value, e.g. Non-Western.")
    ap.add_argument("--mode", default="coherent", choices=["coherent", "drop"],
                    help="Which country-ablation variant to compare against.")
    ap.add_argument("--per-probe-country", action="store_true",
                    help="Instead of the three columns, show agreement between "
                         "the blank-country prediction and EACH of the eight "
                         "cluster-representative probe countries.")
    ap.add_argument("--errors-only", action="store_true",
                    help="Condition on the model's errors, as the US-default "
                         "rate does.")
    args = ap.parse_args()

    scope = (f"group={args.group}" if args.group else "all non-US") + \
            (", errors only" if args.errors_only else ", all examples")
    print(f"NormAd _fs2_yn, no-country mode={args.mode} — {scope}\n")
    if args.per_probe_country:
        for size in MODELS:
            for cond in CONDS:
                r = per_probe_country(cond, size, args.mode, args.group,
                                      args.errors_only)
                if r is None:
                    continue
                print(f"### {size} {cond}   "
                      f"P(probe prediction == blank-country prediction)")
                for slug, (v, n) in sorted(r.items(), key=lambda kv: -kv[1][0]):
                    print(f"  {REPS[slug]:28s} {v:.3f} ± {ci(v, n):.3f}  n={n}")
                print()
        return
    print(f"{'condition':22s} {'real vs US':>16s} {'real vs blank':>16s} "
          f"{'US vs blank':>16s} {'n':>6s}")
    missing = []
    for size in MODELS:
        print(f"--- {size}")
        for cond in CONDS:
            r = rates(cond, size, args.group, args.errors_only, args.mode)
            if r is None:
                missing.append(f"{cond}/{size}")
                continue
            print(f"{cond:22s} "
                  + " ".join(f"{r[k]:9.3f}±{ci(r[k], r['n']):.3f}"
                             for k in ("real_vs_us", "real_vs_blank",
                                       "us_vs_blank"))
                  + f" {r['n']:6d}")
    if missing:
        print(f"\nmissing country-ablation runs: {', '.join(missing)}")
        print("run slurm/eval_normad_nocountry_job.sh (+ the gemma4 variant)")


if __name__ == "__main__":
    main()
