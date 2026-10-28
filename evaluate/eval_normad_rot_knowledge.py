#!/usr/bin/env python
"""Ask about the Rule-of-Thumb with no scenario.

The scenario eval:

    Country: Nepal
    Scenario: Sarah passed a book with her left hand. Is what Sarah did acceptable?
    -> yes / no

This:

    Country: Nepal
    Norm: One must always pass, offer, and receive objects with the right hand.
    Question: Is this a social norm in this country?
    -> yes / no

Same template with "Scenario:" swapped for "Norm:". Same 1,697 examples as the
scenario arm (yes/no only, holdout countries excluded) in the same order, so
the two compare per example.

With --us-probe or --probe-country X the norm is asked again with the country
replaced, keeping the norm fixed. That answer goes in us_pred, the same field
the scenario arm uses.

Scoring only, never generation: one forward pass per prompt, log_softmax over
the final position, batch size 1 -- identical to eval_normad.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    build_chat_prompt,
    NEUTRAL_CHAT_SYSTEM,
    culture_group,
    is_instruct,
    load_model_for_eval,
    resolve_condition,
)
from evaluate.eval_normad import (
    ALL_CONDITIONS,
    HOLDOUT_COUNTRIES,
    country,
    gold_label,
    load_normad,
    rule_of_thumb,
    score_choices,
)

YN_CHOICES = ["yes", "no"]

# A probe country that IS the example's own country makes the swap a no-op, so
# it is skipped. Comparing the two strings raw does not catch it: NormAd spells
# the US "united_states_of_america", country() maps that to "US", and the probe
# is named "United States". The scenario arm hit the same mismatch and left 80
# CulturalBench rows trivially agreeing with themselves, so canonicalise.
_US_ALIASES = {"us", "usa", "u.s.", "u.s.a.", "united_states", "america",
               "united_states_of_america", "united states"}


def _canon(c: str) -> str:
    c = str(c).strip().lower().replace(" ", "_").replace("-", "_")
    return "us" if c in _US_ALIASES else c


def same_country(a: str, b: str) -> bool:
    return _canon(a) == _canon(b)

# The v1 scenario template with "Scenario:" replaced by "Norm:". Every other
# word is identical, which is what makes the two arms comparable.
ROT_PROMPT_TEMPLATE = (
    "You are evaluating cultural knowledge.\n"
    "Country: {country}\n"
    "Norm: {rot}\n"
    "Question: Is this a social norm in this country?\n"
    "Answer with exactly one of: yes, no.\n"
    "Answer:"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    p.add_argument("--model-size", default="8b")
    p.add_argument("--precision", default="matched_bf16")
    p.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "NormAd"))
    p.add_argument("--out-path", default=None)
    p.add_argument("--us-probe", action="store_true",
                   help="Also ask the same norm with the country replaced by the "
                        "United States. Result goes in us_pred, matching the "
                        "scenario arm. Output gains _usprobe.")
    p.add_argument("--probe-country", default=None, metavar="COUNTRY",
                   help="Same as --us-probe with an arbitrary country (its answer "
                        "still lands in us_pred, for compatibility). Output gains "
                        "a _<slug>probe suffix. Overrides --us-probe.")
    p.add_argument("--probe-countries", nargs="+", default=None, metavar="COUNTRY",
                   help="Several probe countries in one run, written to one file "
                        "each. The own-country answers are identical across probes, "
                        "so scoring them once here instead of once per probe job "
                        "costs 1 + k passes per example rather than 2k: for the "
                        "8-country grid that is 9 instead of 16. Use 'us' for the "
                        "United States.")
    p.add_argument("--system-prompt", default=None, metavar="TEXT")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if args.system_prompt == "neutral":
        args.system_prompt = NEUTRAL_CHAT_SYSTEM

    cond = resolve_condition(args.condition, model_size=args.model_size)
    tokenizer, model = load_model_for_eval(cond, precision=args.precision)
    instruct = is_instruct(args.model_size)
    leading_space = not instruct

    ds = load_normad(Path(args.data_path))
    eligible = [ex for ex in ds
                if gold_label(ex) in ("yes", "no")
                and country(ex) not in HOLDOUT_COUNTRIES
                and rule_of_thumb(ex)]
    print(f"Eligible examples: {len(eligible)}")

    # Probe list, as (country-as-written-in-the-prompt, filename slug). "us"
    # becomes "United States" and the _usprobe suffix, matching the scenario arm.
    if args.probe_countries:
        probes = [("United States", "usprobe") if x.lower() == "us"
                  else (x, f"{x.lower().replace(' ', '_')}probe")
                  for x in args.probe_countries]
    elif args.probe_country:
        probes = [(args.probe_country,
                   f"{args.probe_country.lower().replace(' ', '_')}probe")]
    elif args.us_probe:
        probes = [("United States", "usprobe")]
    else:
        probes = []
    if probes:
        print(f"Country swaps: {', '.join(p for p, _ in probes)}")

    def ask(country_name: str, rot: str):
        body = ROT_PROMPT_TEMPLATE.format(country=country_name, rot=rot)
        prompt = (build_chat_prompt(tokenizer, body, system=args.system_prompt)
                  if instruct else body)
        s = score_choices(model, tokenizer, prompt, YN_CHOICES,
                          leading_space=leading_space)
        return YN_CHOICES[max(range(2), key=s.__getitem__)], s

    records = []
    # slug -> per-example swapped answer, aligned with records by index. None
    # where the probe country is the example's own, which is a no-op.
    swapped: dict[str, list[dict | None]] = {slug: [] for _, slug in probes}

    items = eligible[: args.limit] if args.limit else eligible
    for ex in tqdm(items, desc=f"normad_rotknow/{args.condition}"):
        c = country(ex)
        rot = rule_of_thumb(ex)
        pred, scores = ask(c, rot)
        records.append({
            "country": c,
            "group": culture_group(c),
            "gold": gold_label(ex),      # the SCENARIO's gold, for pairing
            "rot": rot,
            "pred": pred,
            "scores": scores,
        })
        for probe_name, slug in probes:
            if same_country(c, probe_name):
                swapped[slug].append(None)
                continue
            u_pred, u_scores = ask(probe_name, rot)
            swapped[slug].append({"us_pred": u_pred, "us_scores": u_scores})

    n = len(records)
    if n == 0:
        raise SystemExit("no scorable examples")
    yes_rate = sum(r["pred"] == "yes" for r in records) / n

    by_group: dict[str, list[bool]] = {}
    by_country: dict[str, list[bool]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r["pred"] == "yes")
        by_country.setdefault(r["country"], []).append(r["pred"] == "yes")

    base = {
        "condition": args.condition,
        "benchmark": "NormAd-norm-only",
        "n": n,
        "yes_rate": yes_rate,
        "yes_rate_by_group": {k: sum(v) / len(v) for k, v in sorted(by_group.items())},
        "yes_rate_by_country": {k: sum(v) / len(v) for k, v in sorted(by_country.items())},
    }
    print(f"  says yes to the norm: {yes_rate:.3f}   n={n}")

    def write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: on the scavenger partition a preemption mid-dump leaves a
        # truncated file that still parses as the start of an object.
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
        print(f"Wrote {path}")

    stem = (PROJECT_ROOT / "outputs" / "behavioral"
            / f"normad_{args.condition}_{args.model_size}_rotknow")

    # One file per probe country, each holding the own-country answers plus
    # that probe's swapped answer -- the same shape a single-probe run wrote,
    # so nothing downstream needs to know this was one job.
    if not probes:
        write(Path(args.out_path) if args.out_path else stem.with_name(stem.name + ".json"),
              {**base, "probe_country": None, "records": records})
        return

    for probe_name, slug in probes:
        rows = []
        for rec, extra in zip(records, swapped[slug]):
            rows.append({**rec, **extra} if extra else dict(rec))
        withp = [r for r in rows if "us_pred" in r]
        write(stem.with_name(f"{stem.name}_{slug}.json"),
              {**base, "probe_country": probe_name, "records": rows})
        if withp:
            agree = sum(r["us_pred"] == r["pred"] for r in withp) / len(withp)
            pyes = sum(r["us_pred"] == "yes" for r in withp) / len(withp)
            print(f"  swapped to {probe_name}: yes {pyes:.3f}, agrees with own "
                  f"country {agree:.3f}   n={len(withp)}")


if __name__ == "__main__":
    main()
