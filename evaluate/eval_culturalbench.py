"""CulturalBench evaluation for one condition.

Each row is a binary True/False cultural knowledge question, reformatted into
"In [country], is [option] [predicate]?" form by data/prep_culturalbench.py.
We score yes/no log-probs and compare to gold (answer=True → "yes", False → "no").

Outputs JSON to outputs/behavioral/culturalbench_{condition}_{model}_nfs[_usprobe].json.
"""
from __future__ import annotations

import argparse
import json
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

PROMPT_SUFFIX = "\nAnswer with yes or no.\nAnswer:"


def country_with_article(country: str) -> str:
    return f"the {country}" if country in THE_COUNTRIES else country


def make_probe_prompt(reformatted: str, country: str, probe: str) -> str:
    src = f"In {country_with_article(country)},"
    dst = f"In {country_with_article(probe)},"
    return reformatted.replace(src, dst, 1)


def build_neutral_fewshot_prefix() -> str:
    return "".join(f"{q} {a}\n\n" for q, a in NEUTRAL_SHOTS)


def build_prompt(prefix: str, reformatted: str, instruct: bool,
                 tokenizer, fewshot_turns) -> str:
    full_q = reformatted + PROMPT_SUFFIX
    if instruct:
        return build_chat_prompt(tokenizer, full_q, fewshot=fewshot_turns)
    return prefix + full_q


@torch.no_grad()
def evaluate_one(condition_name: str, out_path: Path,
                 model_size: str = "8b", precision: str = "matched_bf16",
                 us_probe: bool = False, probe_country: str | None = None):
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)
    instruct = is_instruct(model_size)

    rows = json.loads(DATA_PATH.read_text())
    prefix = build_neutral_fewshot_prefix()
    fewshot_turns = NEUTRAL_SHOTS if instruct else None

    _probe = probe_country or ("United States" if us_probe else None)
    leading_space = not instruct

    correct  = defaultdict(int)
    total    = defaultdict(int)
    predictions = []

    for row in tqdm(rows, desc=f"culturalbench/{condition_name}"):
        reformatted = row.get("reformatted_prompt")
        if not reformatted:
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


ALL_CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult", "sftdpo_aya_nocult"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--model-size", default="8b",
                        choices=["8b", "8b_instruct", "gemma4", "gemma4_instruct"])
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

    out = Path(args.out_path) if args.out_path else (
        BEHAVIORAL / f"culturalbench_{args.condition}{size_sfx}_nfs{probe_sfx}.json"
    )
    evaluate_one(
        args.condition, out,
        model_size=args.model_size,
        precision=args.precision,
        us_probe=args.us_probe,
        probe_country=args.probe_country,
    )


if __name__ == "__main__":
    main()
