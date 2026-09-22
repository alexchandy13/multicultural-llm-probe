"""CulturalBench-Normy evaluation for one condition.

CulturalBench-Normy is CulturalBench reformatted into NormAd-style narrative
stories with Anglo names. The country is NOT embedded in the story — it is
injected via the prompt template, matching NormAd's multi-prompt-word format.

Each story ends with a yes/no question derived from the original CulturalBench
question (e.g. "Is what she did a common practice regarding drinks and paying?").

Outputs JSON to:
  outputs/behavioral/culturalbench_normy_{condition}_{model}_nfs_mpw[_usprobe].json

Usage:
    python evaluate/eval_culturalbench_normy.py --condition base --us-probe
    python evaluate/eval_culturalbench_normy.py --condition base --probe-country taiwan
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
from evaluate.eval_normad import (
    NEUTRAL_SHOTS_NORMAD,
    YN_WORD_CHOICES,
    YN_WORD_PROMPTS,
    build_neutral_fewshot_prefix,
    score_choices,
)

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
DATA_PATH  = PROJECT_ROOT / "data" / "culturalbench_normad.json"

ALL_CONDITIONS = ["base", "sft_aya_cult", "sft_aya_nocult", "sftdpo_aya_cult", "sftdpo_aya_nocult"]


@torch.no_grad()
def evaluate_one(condition_name: str, out_path: Path,
                 model_size: str = "8b", precision: str = "matched_bf16",
                 us_probe: bool = False, probe_country: str | None = None):
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)
    instruct = is_instruct(model_size)

    rows = json.loads(DATA_PATH.read_text())
    prefix = build_neutral_fewshot_prefix(multi_prompt_word=True)
    fewshot_turns = NEUTRAL_SHOTS_NORMAD if instruct else None
    choices = YN_WORD_CHOICES
    leading_space = not instruct

    _probe = probe_country or ("United States" if us_probe else None)

    correct     = defaultdict(int)
    total       = defaultdict(int)
    predictions = []

    for row in tqdm(rows, desc=f"culturalbench_normy/{condition_name}"):
        story = row.get("prompt")
        if not story:
            continue

        country = row["country"]
        gold    = row["gold"]
        group   = culture_group(country)

        acc = [0.0, 0.0]
        for tmpl, pfx in zip(YN_WORD_PROMPTS, prefix):
            if instruct:
                up = build_chat_prompt(
                    tokenizer,
                    tmpl.format(country=country, scenario=story),
                    fewshot=fewshot_turns,
                )
            else:
                up = pfx + tmpl.format(country=country, scenario=story)
            s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
            acc[0] += s[0]; acc[1] += s[1]
        pred = choices[0] if acc[0] > acc[1] else choices[1]

        total[("all", "all")] += 1
        total[("country", country)] += 1
        total[("group", group)] += 1
        if pred == gold:
            correct[("all", "all")] += 1
            correct[("country", country)] += 1
            correct[("group", group)] += 1

        us_pred = us_raw_scores = None
        if _probe and country != _probe:
            acc2 = [0.0, 0.0]
            for tmpl, pfx in zip(YN_WORD_PROMPTS, prefix):
                if instruct:
                    up = build_chat_prompt(
                        tokenizer,
                        tmpl.format(country=_probe, scenario=story),
                        fewshot=fewshot_turns,
                    )
                else:
                    up = pfx + tmpl.format(country=_probe, scenario=story)
                s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
                acc2[0] += s[0]; acc2[1] += s[1]
            us_pred = choices[0] if acc2[0] > acc2[1] else choices[1]
            us_raw_scores = acc2

        predictions.append({
            "data_idx":     row["data_idx"],
            "question_idx": row["question_idx"],
            "country":      country,
            "group":        group,
            "gold":         gold,
            "pred":         pred,
            "us_pred":      us_pred,
            "scores":       acc,
            "us_scores":    us_raw_scores,
        })

    def acc_val(key):
        return correct[key] / total[key] if total[key] else None

    result = {
        "condition": condition_name,
        "benchmark": "CulturalBench-Normy",
        "n": total[("all", "all")],
        "accuracy_overall": acc_val(("all", "all")),
        "accuracy_by_group": {
            "Western":     acc_val(("group", "Western")),
            "Non-Western": acc_val(("group", "Non-Western")),
            "Other":       acc_val(("group", "Other")),
        },
        "accuracy_by_country": {
            c: acc_val(("country", c)) for (_, c) in total if _ == "country"
        },
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}: overall={result['accuracy_overall']:.3f}")


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
        BEHAVIORAL / f"culturalbench_normy_{args.condition}{size_sfx}_nfs_mpw{probe_sfx}.json"
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
