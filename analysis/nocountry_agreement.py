"""Does swapping in the US differ from giving no country at all?

For each condition, three agreement rates over the same examples:

  real vs US        P(pred == us_pred)         -- the usual US-default rate
  real vs blank     P(pred == blank_pred)      -- how much the country matters
  US vs blank       P(us_pred == blank_pred)   -- the one that settles it

If "US vs blank" is near 1.0, then telling the model it is the US and telling
it nothing produce the same answers, and a high US-default rate is evidence
that the model ignores the country -- not that it is anchored on the US. A low
value means the US token does something specific.

--per-probe-country compares the country-free prediction against EACH of the
eight cluster-representative probe countries instead. If it agrees with all of
them equally the model is ignoring the country field; a country that agrees
noticeably more is a genuine default.

WHERE THE COUNTRY-FREE PREDICTION LIVES differs by benchmark, because the
ablations were built differently and the files are already on disk:

  normad         --no-country swaps the prompt, so the run's own `pred` IS the
  blend          country-free answer.
  culturalbench  --no-country scores an EXTRA pass per row, so `pred` is still
                 the with-country answer and the country-free one is nc_pred.

BLANK_FIELD below records that per benchmark rather than papering over it; the
CulturalBench layout also means its no-country file alone carries both columns,
with no join needed for "real vs blank".

    python analysis/nocountry_agreement.py
    python analysis/nocountry_agreement.py --benchmark culturalbench
    python analysis/nocountry_agreement.py --benchmark blend --per-probe-country
    python analysis/nocountry_agreement.py --group Non-Western --errors-only
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

# prefix, the shot/format suffix the probe grid was run at, and the field in
# the no-country file that holds the country-free prediction.
BENCHMARKS = {
    "normad":        {"suffix": "_fs2_yn", "blank_field": "pred",
                      "job": "slurm/eval_normad_nocountry_job.sh"},
    "culturalbench": {"suffix": "_nfs4",   "blank_field": "nc_pred",
                      "job": "slurm/eval_culturalbench_nocountry_job.sh"},
    "blend":         {"suffix": "_nfs",    "blank_field": "pred",
                      "job": "slurm/eval_blend_nocountry_job.sh"},
}

REPS = {"us": "US (English)", "netherlands": "Netherlands (Protestant)",
        "hungary": "Hungary (Catholic)",
        "bosnia_and_herzegovina": "Bosnia (Orthodox)",
        "mexico": "Mexico (Latin America)", "taiwan": "Taiwan (Confucian)",
        "philippines": "Philippines (South Asia)",
        "iran": "Iran (African-Islamic)"}

# NormAd and BLEnD write the US as "US", CulturalBench as "United States".
# Comparing raw left 80 CulturalBench rows in the US column agreeing with
# themselves, so canonicalise before excluding.
_US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united_states", "america",
               "united_states_of_america", "united states"}


def _canon(c: str) -> str:
    c = str(c).strip().lower().replace(" ", "_").replace("-", "_")
    return "us" if c in _US_ALIASES else c


def same_country(a: str, b: str) -> bool:
    return _canon(a) == _canon(b)


def load(name: str):
    p = BEHAVIORAL / name
    return json.loads(p.read_text())["predictions"] if p.exists() else None


def ci(p: float, n: int) -> float:
    return 1.96 * sqrt(max(p * (1 - p), 0) / n) if n else 0.0


def path_for(bench: str, cond: str, size: str, probe: str | None) -> str:
    sfx = BENCHMARKS[bench]["suffix"]
    if probe is None:
        return f"{bench}_{cond}_{size}{sfx}_nocountry.json"
    tail = "_usprobe" if probe == "us" else f"_{probe}probe"
    return f"{bench}_{cond}_{size}{sfx}{tail}.json"


def usable(row: dict, bench: str) -> bool:
    """BLEnD flags rows where the country is the sentence subject and could not
    be removed; those still name a country and must not count as country-free."""
    return row.get("nocountry_applied") is not False


def rates(bench: str, cond: str, size: str, group: str | None, errors_only: bool):
    blank_field = BENCHMARKS[bench]["blank_field"]
    us = load(path_for(bench, cond, size, "us"))
    nc = load(path_for(bench, cond, size, None))
    if us is None or nc is None:
        return None
    if len(us) != len(nc):
        raise ValueError(f"{bench}/{cond}/{size}: {len(us)} usprobe rows vs "
                         f"{len(nc)} nocountry rows -- different filters, "
                         "cannot pair")
    pairs = [(u, c) for u, c in zip(us, nc)
             if u.get("us_pred") is not None
             and not same_country(u.get("country", ""), "us")
             and c.get(blank_field) is not None and usable(c, bench)
             and (group is None or u.get("group") == group)]
    if errors_only:
        pairs = [(u, c) for u, c in pairs if u["pred"] != u["gold"]]
    if not pairs:
        return None
    n = len(pairs)
    return {
        "real_vs_us":    sum(u["pred"] == u["us_pred"] for u, _ in pairs) / n,
        "real_vs_blank": sum(u["pred"] == c[blank_field] for u, c in pairs) / n,
        "us_vs_blank":   sum(u["us_pred"] == c[blank_field] for u, c in pairs) / n,
        "n": n,
    }


def per_probe_country(bench: str, cond: str, size: str, group: str | None,
                      errors_only: bool):
    """{probe country: (P(probe_pred == blank_pred), n)}.

    Each probe file stores its swapped prediction in us_pred regardless of
    which country was swapped in, and leaves it None for rows already from
    that country, so those rows drop out per country and n varies slightly.
    """
    blank_field = BENCHMARKS[bench]["blank_field"]
    nc = load(path_for(bench, cond, size, None))
    if nc is None:
        return None
    out = {}
    for slug in REPS:
        pb = load(path_for(bench, cond, size, slug))
        if pb is None:
            continue
        if len(pb) != len(nc):
            raise ValueError(f"{bench}/{cond}/{size}/{slug}: {len(pb)} probe "
                             f"rows vs {len(nc)} nocountry rows, cannot pair")
        pairs = [(a, b) for a, b in zip(pb, nc)
                 if a.get("us_pred") is not None
                 and b.get(blank_field) is not None and usable(b, bench)
                 and (group is None or a.get("group") == group)]
        if errors_only:
            pairs = [(a, b) for a, b in pairs if a["pred"] != a["gold"]]
        if pairs:
            agree = sum(a["us_pred"] == b[blank_field] for a, b in pairs)
            out[slug] = (agree / len(pairs), len(pairs))
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="normad", choices=list(BENCHMARKS),
                    help="Which benchmark's country ablation to read "
                         "(default normad).")
    ap.add_argument("--group", default=None,
                    help="Restrict to one group field value, e.g. Non-Western.")
    ap.add_argument("--per-probe-country", action="store_true",
                    help="Instead of the three columns, show agreement between "
                         "the blank-country prediction and EACH of the eight "
                         "cluster-representative probe countries.")
    ap.add_argument("--errors-only", action="store_true",
                    help="Condition on the model's errors, as the US-default "
                         "rate does.")
    args = ap.parse_args()

    bench = args.benchmark
    sfx = BENCHMARKS[bench]["suffix"]
    scope = (f"group={args.group}" if args.group else "all non-US") + \
            (", errors only" if args.errors_only else ", all examples")
    print(f"{bench} {sfx}, country-free prompt — {scope}\n")

    if args.per_probe_country:
        any_found = False
        for size in MODELS:
            for cond in CONDS:
                r = per_probe_country(bench, cond, size, args.group,
                                      args.errors_only)
                if r is None:
                    continue
                any_found = True
                print(f"### {size} {cond}   "
                      f"P(probe prediction == blank-country prediction)")
                for slug, (v, n) in sorted(r.items(), key=lambda kv: -kv[1][0]):
                    print(f"  {REPS[slug]:28s} {v:.3f} ± {ci(v, n):.3f}  n={n}")
                print()
        if not any_found:
            print(f"no {bench} country-ablation runs found — "
                  f"run {BENCHMARKS[bench]['job']}")
        return

    print(f"{'condition':22s} {'real vs US':>16s} {'real vs blank':>16s} "
          f"{'US vs blank':>16s} {'n':>6s}")
    missing = []
    for size in MODELS:
        print(f"--- {size}")
        for cond in CONDS:
            r = rates(bench, cond, size, args.group, args.errors_only)
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
        print(f"run {BENCHMARKS[bench]['job']} (+ the gemma4 variant)")


if __name__ == "__main__":
    main()
