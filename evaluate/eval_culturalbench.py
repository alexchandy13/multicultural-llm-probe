"""CulturalBench evaluation for one condition.

Each row is a binary True/False cultural knowledge question, reformatted into
"In [country], is [option] [predicate]?" form by data/prep_culturalbench.py.
We score yes/no log-probs and compare to gold (answer=True → "yes", False → "no").

Outputs JSON to outputs/behavioral/culturalbench_{condition}_{model}_nfs[_usprobe].json.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    build_chat_prompt,
    culture_group,
    is_instruct,
    load_model_for_eval,
    resolve_condition,
)
from evaluate.eval_normad import score_choices

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
DATA_PATH  = PROJECT_ROOT / "data" / "culturalbench_reformatted.json"

# Countries that take "the" as a determiner
THE_COUNTRIES = {
    "Netherlands", "United Kingdom", "United States",
    "United Arab Emirates", "Dominican Republic", "Philippines",
    "Czech Republic",
}

CHOICES = ["yes", "no"]

NEUTRAL_SHOTS = [
    ("Is the Earth round?\nAnswer:", "yes"),
    ("Is fire cold?\nAnswer:", "no"),
]

# Extra neutral shots, for --neutral-shots > 2. Same requirements as the pair
# above: no country is named, so nothing cultural leaks and no example has to be
# held out of eval (n stays 4905, unlike --few-shot's 4861).
NEUTRAL_YES_POOL = [
    ("Is the Earth round?\nAnswer:", "yes"),
    ("Is water a liquid at room temperature?\nAnswer:", "yes"),
    ("Does the sun rise in the east?\nAnswer:", "yes"),
    ("Do birds have feathers?\nAnswer:", "yes"),
]
NEUTRAL_NO_POOL = [
    ("Is fire cold?\nAnswer:", "no"),
    ("Do fish breathe through lungs?\nAnswer:", "no"),
    ("Is the moon larger than the sun?\nAnswer:", "no"),
    ("Does ice sink in water?\nAnswer:", "no"),
    ("Are triangles four-sided?\nAnswer:", "no"),
    ("Is snow hot to the touch?\nAnswer:", "no"),
]

PROMPT_SUFFIX = "\nAnswer with yes or no.\nAnswer:"


def country_with_article(country: str) -> str:
    return f"the {country}" if country in THE_COUNTRIES else country


def make_probe_prompt(reformatted: str, country: str, probe: str) -> str:
    src = f"In {country_with_article(country)},"
    dst = f"In {country_with_article(probe)},"
    return reformatted.replace(src, dst, 1)


def build_neutral_shots(n_shots: int, gold_yes_rate: float, seed: int = 7):
    """Pick n_shots country-free shots whose labels match the dataset's yes-rate.

    The hardcoded 2-shot prefix is 1 yes / 1 no, which signals a 50% yes-rate to
    a model whose gold rate is 27% — pushing predictions toward yes, the direction
    every condition already over-predicts. At n_shots=4 the gold-matched split is
    1 yes / 3 no. Same reasoning as build_fewshot_prefix, minus the leakage, since
    no neutral shot names a country.

    n_shots=2 returns NEUTRAL_SHOTS verbatim so existing _nfs results stay comparable.
    """
    if n_shots == 2:
        return list(NEUTRAL_SHOTS)

    n_yes = min(max(round(n_shots * gold_yes_rate), 1), n_shots - 1)
    n_no = n_shots - n_yes
    if n_yes > len(NEUTRAL_YES_POOL) or n_no > len(NEUTRAL_NO_POOL):
        raise ValueError(
            f"n_shots={n_shots} needs {n_yes} yes / {n_no} no neutral shots, pools "
            f"hold {len(NEUTRAL_YES_POOL)} / {len(NEUTRAL_NO_POOL)}"
        )
    picked = NEUTRAL_YES_POOL[:n_yes] + NEUTRAL_NO_POOL[:n_no]
    random.Random(seed).shuffle(picked)   # don't let the lone yes sit at a fixed position
    return picked


def build_neutral_fewshot_prefix(shots=None) -> str:
    return "".join(f"{q} {a}\n\n" for q, a in (shots or NEUTRAL_SHOTS))


# Shots are drawn only from these countries, and every example from them is
# dropped from evaluation — the same no-leakage design as NormAd's HOLDOUT_COUNTRIES.
# New Zealand: 44 of 4905 examples (0.9%), and Western retains 9 other countries
# (Canada, France, Australia, Netherlands, Germany, Italy, Spain, US, UK) so the
# group stays well represented.
HOLDOUT_COUNTRIES = {"New Zealand"}


def build_fewshot_prefix(rows: list[dict], n_shots: int, seed: int = 42):
    """Build an n_shots prefix from HOLDOUT_COUNTRIES, with labels matching gold.

    Returns (prefix, excluded_keys, turns). excluded_keys covers *every* example
    from the holdout countries, not just the sampled shots, so no evaluated item
    concerns a country the model saw demonstrated.

    Shot labels approximate the dataset's yes-rate rather than being balanced.
    CulturalBench gold is ~27% yes, so a 1:1 prefix would tell the model to expect
    ~50% and push predictions toward yes — the direction every condition already
    over-predicts. At n_shots=4 the gold-matched split is 1 yes / 3 no.
    """
    rng = random.Random(seed)

    usable = [r for r in rows if r.get("reformatted_prompt")]
    gold_yes_rate = sum(bool(r["answer"]) for r in usable) / len(usable)

    pool: dict[bool, list[dict]] = {True: [], False: []}
    excluded = set()
    for r in usable:
        if r["country"] not in HOLDOUT_COUNTRIES:
            continue
        excluded.add((r["data_idx"], r["question_idx"]))
        pool[bool(r["answer"])].append(r)

    # at least one of each label, otherwise the prefix teaches only one answer
    n_yes = min(max(round(n_shots * gold_yes_rate), 1), n_shots - 1)
    want = {True: n_yes, False: n_shots - n_yes}
    for lab, k in want.items():
        if len(pool[lab]) < k:
            raise ValueError(
                f"holdout {sorted(HOLDOUT_COUNTRIES)} has {len(pool[lab])} "
                f"{'yes' if lab else 'no'} examples, need {k}"
            )

    picked = []
    for lab, k in want.items():
        rng.shuffle(pool[lab])
        picked.extend(pool[lab][:k])
    rng.shuffle(picked)

    turns = [(r["reformatted_prompt"] + PROMPT_SUFFIX,
              "yes" if r["answer"] else "no") for r in picked]
    prefix = "".join(f"{q} {a}\n\n" for q, a in turns)
    return prefix, excluded, turns


def build_prompt(prefix: str, reformatted: str, instruct: bool,
                 tokenizer, fewshot_turns) -> str:
    full_q = reformatted + PROMPT_SUFFIX
    if instruct:
        return build_chat_prompt(tokenizer, full_q, fewshot=fewshot_turns)
    return prefix + full_q


@torch.no_grad()
def evaluate_one(condition_name: str, out_path: Path,
                 model_size: str = "8b", precision: str = "matched_bf16",
                 us_probe: bool = False, probe_country: str | None = None,
                 neutral_fewshot: bool = True, few_shot: int = 0,
                 neutral_shots: int = 2):
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)
    instruct = is_instruct(model_size)

    rows = json.loads(DATA_PATH.read_text())
    # Precedence: --few-shot N (real shots) > NFS (default) > --no-fewshot (bare).
    excluded_keys: set = set()
    if few_shot > 0:
        prefix, excluded_keys, turns = build_fewshot_prefix(rows, few_shot)
        fewshot_turns = turns if instruct else None
        print(f"Few-shot: {few_shot} real shots, {len(excluded_keys)} excluded from eval")
    elif neutral_fewshot:
        usable = [r for r in rows if r.get("reformatted_prompt")]
        gold_yes_rate = sum(bool(r["answer"]) for r in usable) / len(usable)
        shots = build_neutral_shots(neutral_shots, gold_yes_rate)
        prefix = build_neutral_fewshot_prefix(shots)
        fewshot_turns = shots if instruct else None
        n_yes = sum(a == "yes" for _, a in shots)
        print(f"Neutral few-shot: {len(shots)} shots, {n_yes} yes / "
              f"{len(shots) - n_yes} no (gold yes-rate {gold_yes_rate:.3f})")
    else:
        prefix = ""
        fewshot_turns = None

    _probe = probe_country or ("United States" if us_probe else None)
    leading_space = not instruct

    correct  = defaultdict(int)
    total    = defaultdict(int)
    predictions = []

    for row in tqdm(rows, desc=f"culturalbench/{condition_name}"):
        reformatted = row.get("reformatted_prompt")
        if not reformatted:
            continue
        if (row["data_idx"], row["question_idx"]) in excluded_keys:
            continue

        country  = row["country"]
        gold     = "yes" if row["answer"] else "no"
        group    = culture_group(country)

        prompt = build_prompt(prefix, reformatted, instruct, tokenizer, fewshot_turns)
        scores = score_choices(model, tokenizer, prompt, CHOICES, leading_space=leading_space)
        pred   = CHOICES[0] if scores[0] > scores[1] else CHOICES[1]

        total[("all", "all")] += 1
        total[("country", country)] += 1
        total[("group", group)] += 1
        if pred == gold:
            correct[("all", "all")] += 1
            correct[("country", country)] += 1
            correct[("group", group)] += 1

        us_pred = us_raw_scores = None
        if _probe and country != _probe:
            probe_q = make_probe_prompt(reformatted, country, _probe)
            probe_prompt = build_prompt(prefix, probe_q, instruct, tokenizer, fewshot_turns)
            us_scores = score_choices(model, tokenizer, probe_prompt, CHOICES, leading_space=leading_space)
            us_pred = CHOICES[0] if us_scores[0] > us_scores[1] else CHOICES[1]
            us_raw_scores = list(us_scores)

        predictions.append({
            "data_idx":   row["data_idx"],
            "question_idx": row["question_idx"],
            "country":    country,
            "group":      group,
            "gold":       gold,
            "pred":       pred,
            "us_pred":    us_pred,
            "scores":     list(scores),
            "us_scores":  us_raw_scores,
        })

    def acc(key):
        return correct[key] / total[key] if total[key] else None

    result = {
        "condition": condition_name,
        "benchmark": "CulturalBench",
        "n": total[("all", "all")],
        "accuracy_overall": acc(("all", "all")),
        "accuracy_by_group": {
            "Western":     acc(("group", "Western")),
            "Non-Western": acc(("group", "Non-Western")),
            "Other":       acc(("group", "Other")),
        },
        "accuracy_by_country": {
            c: acc(("country", c)) for (_, c) in total if _ == "country"
        },
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}: overall={result['accuracy_overall']:.3f}")


ALL_CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult",
                  "sftdpo_aya_cult", "sftdpo_aya_nocult",
                  "tulu3_sft", "tulu3_dpo"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--model-size", default="8b",
                        choices=["8b", "8b_instruct", "gemma4", "gemma4_instruct",
                                 "olmoe", "olmoe_sft", "olmoe_instruct",
                 "gemma4_moe", "gemma4_moe_instruct",
                 "olmo3", "olmo3_sft", "olmo3_dpo"])
    parser.add_argument("--few-shot", type=int, default=0, metavar="N",
                        help="Use N real demonstrations drawn from the data (alternating "
                             "yes/no) instead of the neutral shots. The shots are excluded "
                             "from eval. Output gains a _fsN suffix.")
    parser.add_argument("--neutral-shots", type=int, default=2, metavar="N",
                        help="Number of country-free neutral shots (default 2). Labels are "
                             "gold-matched, so N=4 gives 1 yes / 3 no rather than 2/2. No "
                             "example is excluded from eval. Output suffix becomes _nfsN "
                             "for N != 2.")
    parser.add_argument("--no-fewshot", action="store_true",
                        help="Drop the 2 neutral few-shot examples (the hardcoded default) and evaluate 0-shot. Output loses the _nfs suffix.")
    parser.add_argument("--precision", default="matched_bf16")
    parser.add_argument("--us-probe", action="store_true")
    parser.add_argument("--probe-country", default=None)
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args()

    size_sfx = f"_{args.model_size}" if args.model_size != "8b" else "_8b"
    if args.probe_country:
        slug = args.probe_country.lower().replace(" ", "_")
        probe_sfx = f"_{slug}probe"
    elif args.us_probe:
        probe_sfx = "_usprobe"
    else:
        probe_sfx = ""

    if args.few_shot > 0:
        shot_sfx = f"_fs{args.few_shot}"
    elif args.no_fewshot:
        shot_sfx = ""
    else:
        # _nfs stays bare at the historical 2 shots so old filenames keep matching
        shot_sfx = "_nfs" if args.neutral_shots == 2 else f"_nfs{args.neutral_shots}"

    out = Path(args.out_path) if args.out_path else (
        BEHAVIORAL / f"culturalbench_{args.condition}{size_sfx}{shot_sfx}{probe_sfx}.json"
    )
    evaluate_one(
        args.condition, out,
        model_size=args.model_size,
        precision=args.precision,
        us_probe=args.us_probe,
        probe_country=args.probe_country,
        neutral_fewshot=not args.no_fewshot,
        few_shot=args.few_shot,
        neutral_shots=args.neutral_shots,
    )


if __name__ == "__main__":
    main()
