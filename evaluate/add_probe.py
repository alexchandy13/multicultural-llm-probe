"""Add a cluster-representative probe to an existing NormAd prediction file.

Loads an existing _usprobe.json (which already has pred/scores for the real
country) and runs only the probe-country scoring pass, writing a new file with
us_pred/us_scores filled in for the probe country.

This uses half the GPU memory compared to running eval_normad.py from scratch,
because the model runs only once per example (probe pass only).

Usage:
    python evaluate/add_probe.py \
        --base outputs/behavioral/normad_sftdpo_aya_cult_8b_nfs_mpw_usprobe.json \
        --probe-country iran \
        --condition sftdpo_aya_cult \
        --model-size 8b
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    build_chat_prompt,
    is_instruct,
    load_model_for_eval,
    resolve_condition,
)
from evaluate.eval_normad import (
    NEUTRAL_SHOTS_NORMAD,
    YN_WORD_CHOICES,
    YN_WORD_PROMPTS,
    build_neutral_fewshot_prefix,
    country,
    gold_label,
    load_normad,
    scenario_text,
    score_choices,
    HOLDOUT_COUNTRIES,
)

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="Existing _usprobe.json to read pred/scores from.")
    parser.add_argument("--probe-country", required=True,
                        help="Country name to substitute as probe (e.g. 'iran').")
    parser.add_argument("--condition", required=True,
                        help="Condition name (e.g. sftdpo_aya_cult).")
    parser.add_argument("--model-size", default="8b")
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "normad"))
    parser.add_argument("--out", default=None,
                        help="Output path. Default: auto-named in outputs/behavioral/.")
    parser.add_argument("--precision", default="matched_bf16")
    args = parser.parse_args()

    base_path = Path(args.base)
    base_data = json.loads(base_path.read_text())
    existing_preds = base_data["predictions"]

    probe = args.probe_country
    slug = probe.lower().replace(" ", "_")

    out_path = Path(args.out) if args.out else (
        BEHAVIORAL / re.sub(r"_usprobe.*$", f"_nfs_mpw_{slug}probe.json",
                            base_path.name)
    )

    print(f"Base file : {base_path.name}  ({len(existing_preds)} predictions)")
    print(f"Probe     : {probe}")
    print(f"Output    : {out_path.name}")

    cond = resolve_condition(args.condition, model_size=args.model_size)
    tokenizer, model = load_model_for_eval(cond, precision=args.precision)
    instruct = is_instruct(args.model_size)

    ds = load_normad(Path(args.data_path))

    prefix = build_neutral_fewshot_prefix(multi_prompt_word=True)
    fewshot_turns = NEUTRAL_SHOTS_NORMAD if instruct else None
    choices = YN_WORD_CHOICES

    # Replicate the same exclusion logic as eval_normad with --neutral-fewshot --multi-prompt-word
    holdout_excluded = {i for i, ex in enumerate(ds) if country(ex) in HOLDOUT_COUNTRIES}
    excluded = set()  # neutral_fewshot: no dataset examples excluded beyond yn neutral filter

    pred_iter = iter(existing_preds)
    new_predictions = []

    for i, ex in enumerate(tqdm(ds, desc=f"add_probe/{slug}")):
        if i in holdout_excluded:
            continue
        if gold_label(ex) == "neutral":
            continue

        existing = next(pred_iter)
        c = country(ex)
        leading_space = not instruct

        # Only run the probe pass
        us_pred = None
        us_raw_scores = None
        if c != probe:
            us_accumulated = [0.0, 0.0]
            for tmpl, pfx in zip(YN_WORD_PROMPTS, prefix):
                if instruct:
                    up = build_chat_prompt(
                        tokenizer,
                        tmpl.format(country=probe, scenario=scenario_text(ex)),
                        fewshot=fewshot_turns,
                    )
                else:
                    up = pfx + tmpl.format(country=probe, scenario=scenario_text(ex))
                s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
                us_accumulated[0] += s[0]
                us_accumulated[1] += s[1]
            us_pred = "yes" if us_accumulated[0] > us_accumulated[1] else "no"
            us_raw_scores = us_accumulated

        new_predictions.append({
            "country":   existing["country"],
            "group":     existing["group"],
            "gold":      existing["gold"],
            "pred":      existing["pred"],
            "us_pred":   us_pred,
            "scores":    existing.get("scores"),
            "us_scores": us_raw_scores,
        })

    out_data = dict(base_data)
    out_data["predictions"] = new_predictions
    out_data["probe_country"] = probe

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"Wrote {out_path}  ({len(new_predictions)} predictions)")


if __name__ == "__main__":
    main()
