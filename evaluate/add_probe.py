"""Add a cluster-representative probe to an existing NormAd or BLEnD prediction file.

Loads an existing _usprobe.json (which already has pred/scores for the real
country) and runs only the probe-country scoring pass, writing a new file with
us_pred/us_scores filled in for the probe country.

This uses half the GPU memory compared to running eval_normad/blend.py from scratch,
because the model runs only once per example (probe pass only).

Usage (NormAd):
    python evaluate/add_probe.py \
        --base outputs/behavioral/normad_sftdpo_aya_cult_8b_nfs_mpw_usprobe.json \
        --probe-country iran --condition sftdpo_aya_cult --model-size 8b

Usage (BLEnD):
    python evaluate/add_probe.py \
        --base outputs/behavioral/blend_base_8b_nfs_usprobe.json \
        --probe-country taiwan --condition base --model-size 8b
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    build_chat_prompt,
    is_instruct,
    load_model_for_eval,
    resolve_condition,
)

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"


def run_normad_probe(base_data: dict, probe: str, model, tokenizer,
                     instruct: bool, data_path: Path) -> list[dict]:
    from evaluate.eval_normad import (
        NEUTRAL_SHOTS_NORMAD, YN_WORD_CHOICES, YN_WORD_PROMPTS,
        build_neutral_fewshot_prefix, country, gold_label, load_normad,
        scenario_text, score_choices, HOLDOUT_COUNTRIES,
    )
    ds = load_normad(data_path)
    prefix = build_neutral_fewshot_prefix(multi_prompt_word=True)
    fewshot_turns = NEUTRAL_SHOTS_NORMAD if instruct else None
    choices = YN_WORD_CHOICES
    holdout_excluded = {i for i, ex in enumerate(ds) if country(ex) in HOLDOUT_COUNTRIES}

    pred_iter = iter(base_data["predictions"])
    new_predictions = []
    leading_space = not instruct

    for i, ex in enumerate(tqdm(ds, desc=f"add_probe(normad)/{probe}")):
        if i in holdout_excluded:
            continue
        if gold_label(ex) == "neutral":
            continue
        existing = next(pred_iter)
        c = country(ex)

        us_pred = us_raw_scores = None
        if c != probe:
            acc = [0.0, 0.0]
            for tmpl, pfx in zip(YN_WORD_PROMPTS, prefix):
                if instruct:
                    up = build_chat_prompt(tokenizer, tmpl.format(country=probe, scenario=scenario_text(ex)), fewshot=fewshot_turns)
                else:
                    up = pfx + tmpl.format(country=probe, scenario=scenario_text(ex))
                s = score_choices(model, tokenizer, up, choices, leading_space=leading_space)
                acc[0] += s[0]; acc[1] += s[1]
            us_pred = "yes" if acc[0] > acc[1] else "no"
            us_raw_scores = acc

        new_predictions.append({
            "country": existing["country"], "group": existing["group"],
            "gold": existing["gold"], "pred": existing["pred"],
            "us_pred": us_pred, "scores": existing.get("scores"),
            "us_scores": us_raw_scores,
        })
    return new_predictions


def run_blend_probe(base_data: dict, probe: str, model, tokenizer,
                    instruct: bool, data_path: Path) -> list[dict]:
    from evaluate.eval_blend import (
        NEUTRAL_SHOTS_BLEND, build_neutral_fewshot_prefix, load_blend,
        score_choices, us_probe_prompt, SCORING_SUFFIX,
    )
    ds = load_blend(data_path)
    prefix = build_neutral_fewshot_prefix(multi_prompt=False)
    fewshot_turns = NEUTRAL_SHOTS_BLEND if instruct else None

    pred_iter = iter(base_data["predictions"])
    new_predictions = []

    for ex in tqdm(ds, desc=f"add_probe(blend)/{probe}"):
        existing = next(pred_iter)
        c = ex["country"]

        us_pred = None
        if c != probe:
            prompt = ex["prompt"]
            probe_prompt = us_probe_prompt(prompt, c, probe=probe)
            if instruct:
                up = build_chat_prompt(tokenizer, probe_prompt, fewshot=fewshot_turns, generation_suffix=SCORING_SUFFIX)
            else:
                up = prefix + probe_prompt + SCORING_SUFFIX
            us_scores = score_choices(model, tokenizer, up)
            from evaluate.eval_blend import CHOICES
            us_pred = CHOICES[max(range(4), key=us_scores.__getitem__)]

        new_predictions.append({
            "country": existing["country"], "group": existing["group"],
            "gold": existing["gold"], "pred": existing["pred"],
            "us_pred": us_pred, "mcqid": existing.get("mcqid"),
            "scores": existing.get("scores"),
        })
    return new_predictions


def run_culturalbench_probe(base_data: dict, probe: str, model, tokenizer,
                            instruct: bool) -> list[dict]:
    from evaluate.eval_culturalbench import (
        CHOICES, NEUTRAL_SHOTS, build_neutral_fewshot_prefix,
        build_prompt, make_probe_prompt, DATA_PATH,
    )
    import json
    rows = json.loads(DATA_PATH.read_text())
    prefix = build_neutral_fewshot_prefix()
    fewshot_turns = NEUTRAL_SHOTS if instruct else None
    leading_space = not instruct

    from evaluate.eval_normad import score_choices
    pred_iter = iter(base_data["predictions"])
    new_predictions = []

    for row in tqdm(rows, desc=f"add_probe(culturalbench)/{probe}"):
        if not row.get("reformatted_prompt"):
            continue
        existing = next(pred_iter)
        country = row["country"]

        us_pred = us_raw_scores = None
        if country != probe:
            probe_q = make_probe_prompt(row["reformatted_prompt"], country, probe)
            probe_prompt = build_prompt(prefix, probe_q, instruct, tokenizer, fewshot_turns)
            us_scores = score_choices(model, tokenizer, probe_prompt, CHOICES, leading_space=leading_space)
            us_pred = CHOICES[0] if us_scores[0] > us_scores[1] else CHOICES[1]
            us_raw_scores = list(us_scores)

        new_predictions.append({
            "data_idx":     existing["data_idx"],
            "question_idx": existing["question_idx"],
            "country":      existing["country"],
            "group":        existing["group"],
            "gold":         existing["gold"],
            "pred":         existing["pred"],
            "us_pred":      us_pred,
            "scores":       existing.get("scores"),
            "us_scores":    us_raw_scores,
        })
    return new_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
                        help="Existing _usprobe.json to read pred/scores from.")
    parser.add_argument("--probe-country", required=True,
                        help="Country name to substitute as probe (e.g. 'iran').")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-size", default="8b")
    parser.add_argument("--data-path", default=None,
                        help="Dataset path. Default: data/normad or data/BLEnD based on base filename.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--precision", default="matched_bf16")
    args = parser.parse_args()

    base_path = Path(args.base)
    base_data = json.loads(base_path.read_text())
    probe = args.probe_country
    slug = probe.lower().replace(" ", "_")

    if base_path.name.startswith("blend_"):
        benchmark = "blend"
    elif base_path.name.startswith("culturalbench_"):
        benchmark = "culturalbench"
    else:
        benchmark = "normad"

    out_path = Path(args.out) if args.out else (
        BEHAVIORAL / re.sub(r"_usprobe.*$", f"_{slug}probe.json", base_path.name)
    )

    if args.data_path:
        data_path = Path(args.data_path)
    elif benchmark == "blend":
        data_path = PROJECT_ROOT / "data" / "BLEnD"
    else:
        data_path = PROJECT_ROOT / "data" / "normad"

    print(f"Benchmark : {benchmark}")
    print(f"Base file : {base_path.name}  ({len(base_data['predictions'])} predictions)")
    print(f"Probe     : {probe}")
    print(f"Output    : {out_path.name}")

    cond = resolve_condition(args.condition, model_size=args.model_size)
    tokenizer, model = load_model_for_eval(cond, precision=args.precision)
    instruct = is_instruct(args.model_size)

    if benchmark == "blend":
        new_predictions = run_blend_probe(base_data, probe, model, tokenizer, instruct, data_path)
    elif benchmark == "culturalbench":
        new_predictions = run_culturalbench_probe(base_data, probe, model, tokenizer, instruct)
    else:
        new_predictions = run_normad_probe(base_data, probe, model, tokenizer, instruct, data_path)

    out_data = dict(base_data)
    out_data["predictions"] = new_predictions
    out_data["probe_country"] = probe

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"Wrote {out_path}  ({len(new_predictions)} predictions)")


if __name__ == "__main__":
    main()
