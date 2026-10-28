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

    probe = args.probe_country or ("United States" if args.us_probe else None)
    if probe:
        print(f"Country swap: same norm re-asked as {probe!r} -> us_pred")

    def ask(country_name: str, rot: str):
        body = ROT_PROMPT_TEMPLATE.format(country=country_name, rot=rot)
        prompt = (build_chat_prompt(tokenizer, body, system=args.system_prompt)
                  if instruct else body)
        s = score_choices(model, tokenizer, prompt, YN_CHOICES,
                          leading_space=leading_space)
        return YN_CHOICES[max(range(2), key=s.__getitem__)], s

    records = []
    items = eligible[: args.limit] if args.limit else eligible
    for ex in tqdm(items, desc=f"normad_rotknow/{args.condition}"):
        c = country(ex)
        pred, scores = ask(c, rule_of_thumb(ex))
        rec = {
            "country": c,
            "group": culture_group(c),
            "gold": gold_label(ex),      # the SCENARIO's gold, for pairing
            "rot": rule_of_thumb(ex),
            "pred": pred,
            "scores": scores,
        }
        if probe is not None and c != probe:
            u_pred, u_scores = ask(probe, rule_of_thumb(ex))
            rec["us_pred"] = u_pred
            rec["us_scores"] = u_scores
        records.append(rec)

    n = len(records)
    if n == 0:
        raise SystemExit("no scorable examples")
    yes_rate = sum(r["pred"] == "yes" for r in records) / n

    by_group: dict[str, list[bool]] = {}
    by_country: dict[str, list[bool]] = {}
    for r in records:
        by_group.setdefault(r["group"], []).append(r["pred"] == "yes")
        by_country.setdefault(r["country"], []).append(r["pred"] == "yes")

    result = {
        "condition": args.condition,
        "benchmark": "NormAd-norm-only",
        "n": n,
        "probe_country": probe,
        "yes_rate": yes_rate,
        "yes_rate_by_group": {k: sum(v) / len(v) for k, v in sorted(by_group.items())},
        "yes_rate_by_country": {k: sum(v) / len(v) for k, v in sorted(by_country.items())},
        "records": records,
    }

    if args.probe_country:
        probe_sfx = f"_{args.probe_country.lower().replace(' ', '_')}probe"
    elif args.us_probe:
        probe_sfx = "_usprobe"
    else:
        probe_sfx = ""
    out = Path(args.out_path) if args.out_path else (
        PROJECT_ROOT / "outputs" / "behavioral"
        / f"normad_{args.condition}_{args.model_size}_rotknow{probe_sfx}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: on the scavenger partition a preemption mid-dump leaves a
    # truncated file that still parses as the start of an object.
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out)

    print(f"Wrote {out}")
    print(f"  says yes to the norm: {yes_rate:.3f}   n={n}")
    withp = [r for r in records if "us_pred" in r]
    if withp:
        agree = sum(r["us_pred"] == r["pred"] for r in withp) / len(withp)
        pyes = sum(r["us_pred"] == "yes" for r in withp) / len(withp)
        print(f"  swapped to {probe}: yes {pyes:.3f}, agrees with own country "
              f"{agree:.3f}   n={len(withp)}")


if __name__ == "__main__":
    main()
