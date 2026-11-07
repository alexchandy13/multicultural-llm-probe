"""CulturalBench-Easy evaluation: 4-way MCQ, the format the benchmark ships in.

Why this exists alongside eval_culturalbench.py. That script evaluates
CulturalBench-HARD, which is the same 1,227 questions exploded into 4,908
independent true/false judgements, one per option. We then reformatted each of
those into a yes/no question through an LLM. Two consequences made the result
unreadable:

  * gold_yes is 0.270, so answering "no" to everything scores 0.730. Four of
    the five instruct models landed at or below that line, and the one that
    beat it did so by 0.034. The number the table reported was mostly the
    model's yes-rate.
  * the scored quantity was a yes/no token, so it inherited every surface-form
    problem that a yes/no prompt has -- capitalisation, calibration, the
    few-shot prefix needed to elicit an answer at all.

Easy asks the same cultural question as one 4-way MCQ: 1,227 rows, chance
0.250, and the scored tokens are A/B/C/D, which carry no polarity to be biased
about. It is also the benchmark's own headline format.

The prompt is BLEnD's, down to the JSON answer wrapper, so the two MCQ
benchmarks differ in their questions and nothing else. CHOICES, SCORING_SUFFIX
and score_choices are imported from eval_blend rather than restated -- a second
copy of an MCQ scoring loop is exactly how the probe pass came to score a
different prompt from the baseline on NormAd.

Country swapping is the one thing Easy makes harder than Hard. Hard's questions
were regenerated into a canonical "In [country]," form, so a probe was a string
replace. Easy is in the authors' own words and 69% never write the country
down: "In Korean dining etiquette", "In UK culture", "What do Indians prefer".
evaluate/country_names.py detects whichever form is used and substitutes the
same form of the probe country; 1,092 of 1,227 resolve, and the remaining 135
refer to an ethnic group, a city or a religion rather than a country ("young
Yoruba man", "In Zulu tradition", "In Judaism"). Those are flagged
`swap_applied: false` and the analysis must exclude them -- swapping them would
be wrong, not merely unsupported.

Outputs to outputs/behavioral/cbeasy_{condition}_{model}[...].json. The cbeasy
prefix keeps these clear of the culturalbench_* yes/no files, which stay valid
for what they measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    NEUTRAL_CHAT_SYSTEM,
    build_chat_prompt,
    culture_group,
    is_instruct,
    load_model_for_eval,
    resolve_condition,
)
from evaluate.country_names import (
    resolve_probe_country,
    strip_country_reference,
    strip_unregistered_reference,
    swap_country_reference,
)
# One MCQ implementation, shared with BLEnD.
from evaluate.eval_blend import (
    CHOICES,
    SCORING_SUFFIX,
    _BLEND_INST,
    build_neutral_fewshot_prefix,
    score_choices,
)
from evaluate.eval_blend import NEUTRAL_SHOTS_BLEND

BEHAVIORAL = PROJECT_ROOT / "outputs" / "behavioral"
HF_NAME, HF_CONFIG = "kellycyy/CulturalBench", "CulturalBench-Easy"


def build_prompt(question: str, options: list[str]) -> str:
    """Assemble the MCQ exactly as BLEnD's `prompt` field is assembled.

    BLEnD ships its prompt prebuilt; Easy ships question and options in
    separate columns, so the assembly happens here. _BLEND_INST is imported so
    the instruction wording cannot drift between the two benchmarks.
    """
    body = "\n".join(f"{letter}. {opt}" for letter, opt in zip(CHOICES, options))
    return f"{question}{_BLEND_INST}\n\n{body}\n\nAnswer:"


def options_of(row) -> list[str]:
    return [row[f"prompt_option_{k}"] for k in ("a", "b", "c", "d")]


def permuted(options: list[str], gold: str, key) -> tuple[list[str], str]:
    """Reorder the options deterministically from `key`, returning the new gold.

    CulturalBench-Easy ships 53.6% of its answers as option A, falling
    monotonically to 10.3% for D -- the question writers put the correct option
    first and the distractors after it, and Easy preserved that order. So
    always answering A scores 0.536, and every model here predicts A on about
    half its rows, which means reported accuracy is part cultural knowledge and
    part first-option bias with no way to separate the two.

    The seed is the row's data_idx, so the permutation is identical in the
    baseline, probe and no-country arms. That is not a nicety: probe-vs-baseline
    agreement compares two scorings of the same row, and a permutation that
    differed between them would register as a country effect.
    """
    rng = random.Random(int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16))
    order = list(range(len(CHOICES)))
    rng.shuffle(order)
    return [options[i] for i in order], CHOICES[order.index(CHOICES.index(gold))]


def evaluate_one(condition_name: str, out_path: Path,
                 model_size: str = "8b_instruct",
                 precision: str = "matched_bf16",
                 neutral_fewshot: bool = False,
                 shuffle_options: bool = True,
                 probe_country: str | None = None,
                 no_country: bool = False,
                 raw_prompt: bool = False,
                 system_prompt: str | None = None):
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)

    ds = load_dataset(HF_NAME, HF_CONFIG, split="test")
    countries = sorted(set(ds["country"]))

    # Resolved once, against the dataset's own spellings, so every row's probe
    # prompt names the country identically -- the defect that made the old US
    # column agree with its baseline more than the other seven did.
    probe = resolve_probe_country(probe_country, countries) if probe_country else None
    if probe:
        print(f"Probe country: {probe!r} (from {probe_country!r})")

    instruct = is_instruct(model_size, condition_name) and not raw_prompt
    prefix, fewshot_turns = "", None
    if neutral_fewshot:
        prefix = build_neutral_fewshot_prefix()
        fewshot_turns = list(NEUTRAL_SHOTS_BLEND) if instruct else None
        print(f"Neutral few-shot: {len(NEUTRAL_SHOTS_BLEND)} MCQ-format shots")
    else:
        print("Zero-shot")

    def wrap(text: str) -> str:
        if instruct:
            return build_chat_prompt(tokenizer, text, fewshot=fewshot_turns,
                                     generation_suffix=SCORING_SUFFIX,
                                     system=system_prompt)
        return prefix + text + SCORING_SUFFIX

    correct, total = defaultdict(int), defaultdict(int)
    predictions = []
    swap_counts = defaultdict(int)
    # Printed at the end: the whole point of shuffling is that this comes out
    # flat, so the log should show it rather than leave it to be trusted.
    gold_counts = defaultdict(int)

    for row in tqdm(ds, desc=f"cbeasy/{condition_name}"):
        country = row["country"]
        gold = row["answer"].strip().upper()
        group = culture_group(country)
        question = row["prompt_question"]
        opts = options_of(row)
        if shuffle_options:
            opts, gold = permuted(opts, gold, row["data_idx"])
        gold_counts[gold] += 1

        nc_applied = False
        if no_country:
            # adjective/attributive/name/alias are deleted, plural is
            # replaced with 'people' -- see strip_country_reference(). Rows
            # with no country reference at all fall through to
            # strip_unregistered_reference(), which deletes a curated list of
            # ethnic/religious/regional references ('Yoruba', 'Judaism',
            # 'rainbow nation') that aren't swappable to a country but are
            # still safe to delete. What's left after both (benchmark typos,
            # city/region names) stays excluded.
            question, nc_applied = strip_country_reference(question, country)
            if not nc_applied:
                question, nc_applied = strip_unregistered_reference(question)

        scores = score_choices(model, tokenizer, wrap(build_prompt(question, opts)))
        pred = CHOICES[max(range(4), key=scores.__getitem__)]

        total[("all", "all")] += 1
        total[("country", country)] += 1
        total[("group", group)] += 1
        if pred == gold:
            for key in (("all", "all"), ("country", country), ("group", group)):
                correct[key] += 1

        us_pred, us_scores, swap_kind = None, None, None
        if probe and country != probe:
            swapped, swap_kind = swap_country_reference(question, country, probe)
            if swap_kind is not None:
                us_scores = score_choices(model, tokenizer,
                                          wrap(build_prompt(swapped, opts)))
                us_pred = CHOICES[max(range(4), key=us_scores.__getitem__)]
        swap_counts[swap_kind if probe else "no probe"] += 1

        rec = {
            "data_idx":     row["data_idx"],
            "question_idx": row["question_idx"],
            "country":      country,
            "group":        group,
            "gold":         gold,
            "pred":         pred,
            "scores":       list(scores),
        }
        if probe:
            rec["us_pred"] = us_pred
            rec["us_scores"] = us_scores
            # False when the question names no country, so nothing was
            # substituted and the row measures nothing about the probe.
            rec["swap_applied"] = swap_kind is not None
            rec["swap_form"] = swap_kind
        if no_country:
            rec["nocountry_applied"] = nc_applied
        predictions.append(rec)

    def acc(key):
        return correct[key] / total[key] if total[key] else None

    result = {
        "condition": condition_name,
        "benchmark": "CulturalBench-Easy",
        "n": total[("all", "all")],
        "accuracy_overall": acc(("all", "all")),
        "accuracy_by_group": {
            g: acc(("group", g)) for g in ("Western", "Non-Western", "Other")
        },
        "accuracy_by_country": {
            c: acc(("country", c)) for (kind, c) in total if kind == "country"
        },
        "predictions": predictions,
    }
    if probe:
        result["probe_country"] = probe
        result["probe_n_scored"] = sum(1 for p in predictions if p["us_pred"] is not None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    n = total[("all", "all")]
    result["gold_letter_rate"] = {k: gold_counts[k] / n for k in CHOICES}
    result["options_shuffled"] = shuffle_options
    print(f"Wrote {out_path}: overall={result['accuracy_overall']:.3f}")
    print("  gold letters: " + " ".join(
        f"{k}:{gold_counts[k] / n:.3f}" for k in CHOICES)
        + ("" if shuffle_options else "   (UNSHUFFLED: always-A scores 0.536)"))
    if probe:
        print("  swap forms: " + ", ".join(
            f"{k}={v}" for k, v in sorted(swap_counts.items(), key=lambda kv: str(kv[0]))))


ALL_CONDITIONS = ["base", "offshelf", "dpo", "sft", "sftdpo",
                  "dpo_coig", "dpo_pku", "sftdpo_coig", "sftdpo_pku",
                  "tulu3_sft", "tulu3_dpo",
                  "dpo_cult", "dpo_nocult",
                  "sft_aya_cult", "sft_aya_nocult",
                  "sftdpo_aya_cult", "sftdpo_aya_nocult"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    p.add_argument("--model-size", default="8b_instruct")
    p.add_argument("--precision", default="matched_bf16")
    p.add_argument("--neutral-fewshot", action="store_true",
                   help="Prepend BLEnD's 4 country-free MCQ-format shots.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Keep the dataset's option order. Only for reproducing "
                        "the un-permuted numbers; 53.6%% of golds are A, so "
                        "accuracy then mixes knowledge with first-option bias.")
    p.add_argument("--probe-country", default=None,
                   help="Replace each question's country reference with this "
                        "one and rescore, writing us_pred/us_scores.")
    p.add_argument("--no-country", action="store_true",
                   help="Delete the country reference where it is deletable.")
    p.add_argument("--raw-prompt", action="store_true",
                   help="Bypass the chat template, scoring an instruct "
                        "checkpoint byte-identically to a base model.")
    p.add_argument("--system-prompt", default=None,
                   help="System message for the chat template; 'neutral' is "
                        f"{NEUTRAL_CHAT_SYSTEM!r}.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.system_prompt == "neutral":
        args.system_prompt = NEUTRAL_CHAT_SYSTEM
    if args.no_country and args.probe_country:
        p.error("--no-country and --probe-country are different arms; run them separately")

    # Suffixes are assembled in the same order the other eval scripts use, so
    # the job scripts' out_for() can predict the name.
    nfs = "_nfs" if args.neutral_fewshot else ""
    # Shuffling is the default, so it carries no suffix; the un-permuted run is
    # the one that has to be labelled, since its accuracy is not comparable.
    shuf = "_noshuf" if args.no_shuffle else ""
    probe_sfx = f"_probe_{args.probe_country}" if args.probe_country else ""
    nc = "_nocountry" if args.no_country else ""
    raw = "_raw" if args.raw_prompt and is_instruct(args.model_size, args.condition) else ""
    sys_sfx = "_sys" if args.system_prompt else ""
    out = Path(args.out) if args.out else (
        BEHAVIORAL / f"cbeasy_{args.condition}_{args.model_size}"
                     f"{nfs}{shuf}{probe_sfx}{nc}{raw}{sys_sfx}.json")

    evaluate_one(args.condition, out,
                 model_size=args.model_size, precision=args.precision,
                 neutral_fewshot=args.neutral_fewshot,
                 shuffle_options=not args.no_shuffle,
                 probe_country=args.probe_country,
                 no_country=args.no_country,
                 raw_prompt=args.raw_prompt,
                 system_prompt=args.system_prompt)


if __name__ == "__main__":
    main()
