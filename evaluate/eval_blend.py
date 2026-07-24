"""BLEnD multiple-choice evaluation for one condition.

BLEnD (Benchmark of LLMs Evaluating everyday behaviors across Diverse cultures)
presents multiple-choice questions about cultural practices in 16 countries.
We score options (A/B/C/D) by the model's log-prob of producing that letter
after appending '{"answer_choice":"' to the prompt, with leading_space=False
since the letter follows immediately after the quote character.

Chat templates are never applied — prompts are tokenized as raw text to ensure
all conditions (base/sft/dpo/sftdpo) see identical input formatting.

Outputs JSON to outputs/behavioral/blend_{condition}{suffixes}.json
"""
from __future__ import annotations

import argparse
import json
import random as _random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    culture_group,
    load_model_for_eval,
    resolve_condition,
)


CHOICES = ["A", "B", "C", "D"]
SCORING_SUFFIX = '{"answer_choice":"'

# Countries held out entirely from evaluation; used as the few-shot pool.
# Chosen to be diverse and non-Western.
HOLDOUT_COUNTRIES = {"Algeria", "North_Korea"}

# Inline instruction that mirrors the BLEnD prompt format exactly:
# "[question]? Without any explanation...\n\nA. opt\n...\n\nAnswer:"
_BLEND_INST = (
    ' Without any explanation, choose only one from the given alphabet choices'
    '(e.g., A, B, C). Provide as JSON format: {"answer_choice":""}'
)

# Culturally-agnostic few-shot shots for --neutral-fewshot mode.
# 4 examples covering each answer letter exactly once; none require cultural
# knowledge, so the prefix teaches only the MCQ format and JSON output style.
# Format mirrors the BLEnD prompt field exactly (instruction embedded after '?').
NEUTRAL_SHOTS_BLEND = [
    # gold=A
    ("Which planet is farthest from the Sun in our solar system?" + _BLEND_INST +
     "\n\nA. Neptune\nB. Jupiter\nC. Saturn\nD. Uranus\n\nAnswer:", "A"),
    # gold=B
    ("What is the chemical symbol for water?" + _BLEND_INST +
     "\n\nA. O2\nB. H2O\nC. CO2\nD. NaCl\n\nAnswer:", "B"),
    # gold=C
    ("How many sides does a hexagon have?" + _BLEND_INST +
     "\n\nA. Five\nB. Seven\nC. Six\nD. Eight\n\nAnswer:", "C"),
    # gold=D
    ("What is the largest ocean on Earth?" + _BLEND_INST +
     "\n\nA. Atlantic\nB. Indian\nC. Arctic\nD. Pacific\n\nAnswer:", "D"),
]

# 4 instruction prefixes for multi-prompt mode.
# BLEnD prompts are pre-formatted by the dataset (instruction already embedded);
# these brief wrappers are prepended before each example in the prompt chain.
BLEND_MP_PREFIXES = [
    "Read the following multiple-choice question and select the best answer.\n\n",
    "Answer the following question by choosing the most appropriate option.\n\n",
    "Select the best answer for the following cultural knowledge question.\n\n",
    "Choose the correct answer to the following question about cultural practices.\n\n",
]


def load_blend(data_path: Path):
    """Load BLEnD MCQ test split, from disk cache if available."""
    has_data = data_path.exists() and any(
        p for p in data_path.iterdir() if p.name != ".gitkeep"
    )
    if has_data:
        ds = load_from_disk(str(data_path))
    else:
        ds = load_dataset("nayeon212/BLEnD", "multiple-choice-questions", split="test")
    # Mirror upstream deduplication: cap at 5 MCQ variants per (country, question ID).
    country_id_mcqids: dict[tuple, list] = {}
    for item in ds:
        key = (item["country"], item["ID"])
        if key not in country_id_mcqids:
            country_id_mcqids[key] = []
        if len(country_id_mcqids[key]) < 5:
            country_id_mcqids[key].append(item["MCQID"])
    valid_mcqids = {m for ids in country_id_mcqids.values() for m in ids}
    return ds.filter(lambda x: x["MCQID"] in valid_mcqids)


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str) -> list[float]:
    """Return log-prob of first token of each A/B/C/D choice given prompt.

    leading_space=False: letter token follows immediately after the quote in
    '{"answer_choice":"', so no space prefix is used.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    last_logits = out.logits[0, -1, :].float().log_softmax(-1)

    scores = []
    for choice in CHOICES:
        ids = tokenizer.encode(choice, add_special_tokens=False)
        scores.append(last_logits[ids[0]].item() if ids else float("-inf"))
    return scores


def us_probe_prompt(prompt: str, country: str) -> str:
    """Replace country name in prompt with 'US' for US-probe scoring."""
    replaced = prompt.replace(country, "US")
    if replaced == prompt and "_" in country:
        replaced = prompt.replace(country.replace("_", " "), "US")
    return replaced


def build_neutral_fewshot_prefix(multi_prompt: bool = False) -> str | list[str]:
    """Build a few-shot prefix from NEUTRAL_SHOTS_BLEND.

    Uses 4 culturally-agnostic MCQ examples (one per answer letter A/B/C/D)
    so the prefix teaches only task format, not cultural knowledge.
    No holdout exclusion needed — all 16 countries remain in evaluation.
    """
    if multi_prompt:
        prefixes = []
        for pfx in BLEND_MP_PREFIXES:
            parts = [pfx + q + SCORING_SUFFIX + lbl + '"\n\n'
                     for q, lbl in NEUTRAL_SHOTS_BLEND]
            prefixes.append("".join(parts))
        return prefixes

    return "".join(q + SCORING_SUFFIX + lbl + '"\n\n' for q, lbl in NEUTRAL_SHOTS_BLEND)


def build_fewshot_prefix(ds, n_shots: int, seed: int = 42,
                         multi_prompt: bool = False) -> tuple[str | list[str], set]:
    """Build few-shot prefix from holdout countries.

    Picks n_shots examples from HOLDOUT_COUNTRIES, favouring one per country.
    Returns (prefix, excluded_mcqids) where excluded_mcqids is the full set of
    MCQIDs from all holdout examples (not just the sampled shots).
    """
    rng = _random.Random(seed)
    excluded_mcqids: set = set()
    by_country: dict[str, list] = defaultdict(list)

    for ex in ds:
        if ex["country"] in HOLDOUT_COUNTRIES:
            excluded_mcqids.add(ex["MCQID"])
            by_country[ex["country"]].append(ex)

    if not by_country:
        raise RuntimeError(
            f"Holdout pool is empty. Check HOLDOUT_COUNTRIES {HOLDOUT_COUNTRIES} "
            f"match dataset country field values."
        )

    all_pool = [ex for exs in by_country.values() for ex in exs]
    rng.shuffle(all_pool)

    # Pick shots with distinct gold labels so the model isn't taught to repeat
    # a single letter (analogous to NormAd's forced 1-yes + 1-no balance).
    picks: list[dict] = []
    used_labels: set[str] = set()
    for ex in all_pool:
        if ex["answer_idx"] not in used_labels:
            picks.append(ex)
            used_labels.add(ex["answer_idx"])
            if len(picks) >= n_shots:
                break
    # Fallback: fill remaining slots without label constraint
    for ex in all_pool:
        if len(picks) >= n_shots:
            break
        if ex not in picks:
            picks.append(ex)

    if multi_prompt:
        prefixes = []
        for pfx in BLEND_MP_PREFIXES:
            parts = []
            for ex in picks:
                parts.append(pfx + ex["prompt"] + SCORING_SUFFIX + ex["answer_idx"] + '"\n\n')
            prefixes.append("".join(parts))
        return prefixes, excluded_mcqids

    parts = []
    for ex in picks:
        parts.append(ex["prompt"] + SCORING_SUFFIX + ex["answer_idx"] + '"\n\n')
    return "".join(parts), excluded_mcqids


def evaluate_one(condition_name: str, data_path: Path, out_path: Path,
                 model_size: str = "3b", precision: str = "matched_bf16",
                 few_shot: int = 0, us_probe: bool = False, multi_prompt: bool = False,
                 neutral_fewshot: bool = False):
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)
    ds = load_blend(data_path)

    holdout_excluded = {ex["MCQID"] for ex in ds if ex["country"] in HOLDOUT_COUNTRIES}
    print(f"Holdout excluded: {sorted(HOLDOUT_COUNTRIES)} ({len(holdout_excluded)} examples)")

    if neutral_fewshot:
        prefix = build_neutral_fewshot_prefix(multi_prompt=multi_prompt)
        excluded_mcqids: set = set()  # no dataset examples used; evaluate all 16 countries
        print("Neutral few-shot: 4 culturally-agnostic examples (A/B/C/D each once), full eval set")
    elif few_shot > 0:
        prefix, excluded_mcqids = build_fewshot_prefix(ds, few_shot, multi_prompt=multi_prompt)
        excluded_mcqids |= holdout_excluded
        print(f"Few-shot: {few_shot} examples from holdout pool, {len(excluded_mcqids)} total excluded")
    else:
        prefix = [""] * len(BLEND_MP_PREFIXES) if multi_prompt else ""
        excluded_mcqids = holdout_excluded

    correct: dict = defaultdict(int)
    total: dict = defaultdict(int)
    predictions = []

    for ex in tqdm(ds, desc=f"blend/{condition_name}"):
        if ex["MCQID"] in excluded_mcqids:
            continue

        c = ex["country"]
        gold = ex["answer_idx"]
        prompt = ex["prompt"]
        group = culture_group(c)

        if multi_prompt:
            accumulated = [0.0, 0.0, 0.0, 0.0]
            for pfx_str, pfx_tmpl in zip(prefix, BLEND_MP_PREFIXES):
                p = pfx_str + pfx_tmpl + prompt + SCORING_SUFFIX
                s = score_choices(model, tokenizer, p)
                for j, sj in enumerate(s):
                    accumulated[j] += sj
            pred = CHOICES[max(range(4), key=accumulated.__getitem__)]
        else:
            p = prefix + prompt + SCORING_SUFFIX
            scores = score_choices(model, tokenizer, p)
            pred = CHOICES[max(range(4), key=scores.__getitem__)]

        total[("all", "all")] += 1
        total[("country", c)] += 1
        total[("group", group)] += 1
        if pred == gold:
            correct[("all", "all")] += 1
            correct[("country", c)] += 1
            correct[("group", group)] += 1

        us_pred = None
        if us_probe and c != "US":
            us_prompt = us_probe_prompt(prompt, c)
            if multi_prompt:
                us_acc = [0.0, 0.0, 0.0, 0.0]
                for pfx_str, pfx_tmpl in zip(prefix, BLEND_MP_PREFIXES):
                    up = pfx_str + pfx_tmpl + us_prompt + SCORING_SUFFIX
                    us_s = score_choices(model, tokenizer, up)
                    for j, sj in enumerate(us_s):
                        us_acc[j] += sj
                us_pred = CHOICES[max(range(4), key=us_acc.__getitem__)]
            else:
                up = prefix + us_prompt + SCORING_SUFFIX
                us_scores = score_choices(model, tokenizer, up)
                us_pred = CHOICES[max(range(4), key=us_scores.__getitem__)]

        predictions.append({
            "country": c, "group": group, "gold": gold,
            "pred": pred, "us_pred": us_pred, "mcqid": ex["MCQID"],
        })

    def acc(key):
        return correct[key] / total[key] if total[key] else None

    result = {
        "condition": condition_name,
        "benchmark": "BLEnD",
        "n": total[("all", "all")],
        "accuracy_overall": acc(("all", "all")),
        "accuracy_by_group": {
            "Western": acc(("group", "Western")),
            "Non-Western": acc(("group", "Non-Western")),
            "Other": acc(("group", "Other")),
        },
        "accuracy_by_country": {
            c: acc(("country", c)) for (kind, c) in total if kind == "country"
        },
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}: overall={result['accuracy_overall']:.3f}")


ALL_CONDITIONS = ["base", "dpo", "sft", "sftdpo",
                  "dpo_coig", "dpo_pku", "sftdpo_coig", "sftdpo_pku"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument(
        "--model-size", default="3b",
        choices=["3b", "8b", "8b_instruct", "gemma4", "qwen35"],
    )
    parser.add_argument(
        "--precision", default="matched_bf16",
        choices=["matched_bf16", "qlora_4bit"],
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "data" / "BLEnD"),
    )
    parser.add_argument("--out-path", default=None)
    parser.add_argument(
        "--few-shot", type=int, default=0, metavar="N",
        help="Prepend N few-shot examples from holdout countries (Algeria, North_Korea). "
             "0 = zero-shot (default). Output gains a _fsN suffix.",
    )
    parser.add_argument(
        "--multi-prompt", action="store_true",
        help="Score using 4 instruction-prefix variants; average log-probs before argmax. "
             "Output gains a _mp suffix.",
    )
    parser.add_argument(
        "--us-probe", action="store_true",
        help="For each non-US example, also score with country name replaced by 'US'. "
             "Records us_pred in each prediction entry. Output gains a _usprobe suffix.",
    )
    parser.add_argument(
        "--neutral-fewshot", action="store_true",
        help="Prepend 4 culturally-agnostic MCQ examples (one per answer letter A/B/C/D) "
             "to teach task format without injecting cultural knowledge. No holdout "
             "exclusion — all 16 countries are evaluated. Mutually exclusive with "
             "--few-shot. Output gains a _nfs suffix.",
    )
    args = parser.parse_args()

    if args.neutral_fewshot and args.few_shot > 0:
        import sys; sys.exit("--neutral-fewshot and --few-shot are mutually exclusive")

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    fs_sfx = f"_fs{args.few_shot}" if args.few_shot > 0 else ""
    nfs_sfx = "_nfs" if args.neutral_fewshot else ""
    mp_sfx = "_mp" if args.multi_prompt else ""
    usprobe_sfx = "_usprobe" if args.us_probe else ""

    out = Path(args.out_path) if args.out_path else (
        PROJECT_ROOT / "outputs" / "behavioral"
        / f"blend_{args.condition}{size_sfx}{fs_sfx}{nfs_sfx}{mp_sfx}{usprobe_sfx}.json"
    )
    evaluate_one(
        args.condition,
        Path(args.data_path),
        out,
        model_size=args.model_size,
        precision=args.precision,
        few_shot=args.few_shot,
        us_probe=args.us_probe,
        multi_prompt=args.multi_prompt,
        neutral_fewshot=args.neutral_fewshot,
    )


if __name__ == "__main__":
    main()
