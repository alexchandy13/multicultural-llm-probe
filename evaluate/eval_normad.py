"""NormAd multiple-choice evaluation for one condition.

NormAd scenarios ask whether an action is socially acceptable in a named culture; the
answer key is yes/no/neither. We score each option by the model's log-prob of producing
that token after the prompt, take argmax, and compare to gold.

Outputs JSON to outputs/behavioral/normad_{condition}.json with per-culture breakdown.
"""
from __future__ import annotations

import argparse
import json
import re
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


# Countries held out entirely from evaluation and used as the few-shot pool.
# One per IW cluster (most niche by example count), excluding EnglishSpeaking
# and Confucian clusters so the prefix draws only from non-Western-adjacent
# cultural contexts.
#   AfricanIslamic  → Syria           (n=12)
#   SouthAsia       → Indonesia       (n=18)
#   LatinAmerica    → Colombia        (n=21)
#   CatholicEurope  → Austria         (n=23)
#   Orthodox        → north_macedonia (n=23)
#   ProtestantEurope→ Sweden          (n=24)
HOLDOUT_COUNTRIES = {
    "Syria", "Indonesia", "Colombia",
    "Austria", "north_macedonia", "Sweden",
}

CHOICES = ["yes", "no", "neutral"]

PROMPT_TEMPLATE = (
    "You are evaluating cultural acceptability.\n"
    "Country: {country}\n"
    "Scenario: {scenario}\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "Answer with exactly one of: yes, no, neutral.\n"
    "Answer:"
)

YN_CHOICES = ["yes", "no"]

YN_PROMPT_TEMPLATE = (
    "You are evaluating cultural acceptability.\n"
    "Country: {country}\n"
    "Scenario: {scenario}\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "Answer with exactly one of: yes, no.\n"
    "Answer:"
)

YN_NULL_PROMPT = (
    "You are evaluating cultural acceptability.\n"
    "Country: N/A\n"
    "Scenario: N/A\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "Answer with exactly one of: yes, no.\n"
    "Answer:"
)

# Multiple-choice letter format (Hendrycks et al. 2021 / MMLU style).
# Scoring single letters avoids the surface-form competition problem (Holtzman
# et al. 2021, arXiv 2104.08315) where "neutral" competes with all 'n'-initial
# tokens and is systematically underscored relative to short words like "yes".
MC_CHOICES = ["A", "B", "C"]
MC_LABEL_MAP = {"yes": "A", "no": "B", "neutral": "C"}
MC_CHOICE_MAP = {"A": "yes", "B": "no", "C": "neutral"}

MC_PROMPT_TEMPLATE = (
    "You are evaluating cultural acceptability.\n"
    "Country: {country}\n"
    "Scenario: {scenario}\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "(A) yes  (B) no  (C) neutral\n"
    "Answer:"
)

MC_NULL_PROMPT = (
    "You are evaluating cultural acceptability.\n"
    "Country: N/A\n"
    "Scenario: N/A\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "(A) yes  (B) no  (C) neutral\n"
    "Answer:"
)

# NormAd ships country names lowercase + underscored (e.g. 'united_states_of_america',
# 'south_korea'). Map raw -> canonical forms `culture_group` recognizes; anything
# unmapped falls through and lands in the "Other" bucket. The expanded list covers
# ~50 of NormAd's 75 countries — see evaluate/_common.py for grouping rationale.
NORMAD_COUNTRY_MAP = {
    # Western
    "united_states_of_america": "US",
    "united_kingdom": "UK",
    "germany": "Germany",
    "spain": "Spain",
    "australia": "Australia",
    "canada": "Canada",
    "ireland": "Ireland",
    "new_zealand": "New_Zealand",
    "france": "France",
    "italy": "Italy",
    "netherlands": "Netherlands",
    "austria": "Austria",
    "sweden": "Sweden",
    "portugal": "Portugal",
    "greece": "Greece",
    # Non-Western — East/South/SE Asia
    "japan": "Japan",
    "china": "China",
    "india": "India",
    "indonesia": "Indonesia",
    "south_korea": "South_Korea",
    "pakistan": "Pakistan",
    "bangladesh": "Bangladesh",
    "sri_lanka": "Sri_Lanka",
    "nepal": "Nepal",
    "afghanistan": "Afghanistan",
    "thailand": "Thailand",
    "vietnam": "Vietnam",
    "philippines": "Philippines",
    "malaysia": "Malaysia",
    "singapore": "Singapore",
    "cambodia": "Cambodia",
    "laos": "Laos",
    "myanmar": "Myanmar",
    "hong_kong": "Hong_Kong",
    "taiwan": "Taiwan",
    # Non-Western — MENA + Sub-Saharan Africa
    "iran": "Iran",
    "egypt": "Egypt",
    "lebanon": "Lebanon",
    "iraq": "Iraq",
    "syria": "Syria",
    "saudi_arabia": "Saudi_Arabia",
    "ethiopia": "Ethiopia",
    "kenya": "Kenya",
    "south_africa": "South_Africa",
    "nigeria": "Nigeria",
    # Non-Western — Latin America
    "mexico": "Mexico",
    "brazil": "Brazil",
    "argentina": "Argentina",
    "chile": "Chile",
    "colombia": "Colombia",
    "peru": "Peru",
}


def load_normad(path: Path):
    if path.exists() and any(path.iterdir()):
        ds = load_from_disk(str(path))
    else:
        ds = load_dataset("akhilayerukola/NormAd")
    # Both branches may return a DatasetDict; pick the best split.
    if hasattr(ds, "keys"):
        for split in ("test", "validation", "train"):
            if split in ds:
                return ds[split]
        raise ValueError(f"no usable split in {list(ds.keys())}")
    return ds


def build_fewshot_prefix(ds, n_shots: int, seed: int = 42,
                         mc_format: bool = False,
                         yn_only: bool = False) -> tuple[str, set[int]]:
    """Build a 1-yes + 1-no few-shot prefix from held-out countries.

    Examples are drawn exclusively from HOLDOUT_COUNTRIES — one country per IW
    cluster that is excluded entirely from evaluation. This prevents country-level
    leakage: no held-out country appears in the eval set, and no eval country
    appears in the few-shot context.

    Returns (prefix_string, excluded_indices). excluded_indices contains ALL
    examples from holdout countries (not just the 2 sampled shots), so they
    are never evaluated on.
    """
    import random
    rng = random.Random(seed)

    yes_pool: list[int] = []
    no_pool: list[int] = []
    excluded: set[int] = set()

    for i, ex in enumerate(ds):
        try:
            c = country(ex)
            if c not in HOLDOUT_COUNTRIES:
                continue
            excluded.add(i)
            lbl = gold_label(ex)
            if lbl == "yes":
                yes_pool.append(i)
            elif lbl == "no":
                no_pool.append(i)
        except KeyError:
            pass

    if not yes_pool or not no_pool:
        raise RuntimeError(
            f"Holdout pool missing yes or no examples. yes={len(yes_pool)} no={len(no_pool)}. "
            f"Check that HOLDOUT_COUNTRIES appear in the dataset."
        )

    picks = [rng.choice(yes_pool), rng.choice(no_pool)]
    rng.shuffle(picks)

    if yn_only:
        template = YN_PROMPT_TEMPLATE
    elif mc_format:
        template = MC_PROMPT_TEMPLATE
    else:
        template = PROMPT_TEMPLATE

    parts = []
    for idx in picks:
        ex = ds[idx]
        lbl = gold_label(ex)
        answer = MC_LABEL_MAP[lbl] if mc_format else lbl
        parts.append(template.format(country=country(ex), scenario=scenario_text(ex)) + f" {answer}\n\n")

    return "".join(parts), excluded


NULL_PROMPT = (
    "You are evaluating cultural acceptability.\n"
    "Country: N/A\n"
    "Scenario: N/A\n"
    "Question: Is this action socially acceptable in this culture?\n"
    "Answer with exactly one of: yes, no, neutral.\n"
    "Answer:"
)


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str, choices: list[str],
                  priors: list[float] | None = None) -> list[float]:
    """Return log-prob of the first token of each choice given prompt.

    If `priors` is provided (list of log-probs from a content-free prompt),
    subtract them from each score before returning — this is contextual
    calibration (Zhao et al. 2021) and removes the model's unconditional
    token bias toward e.g. 'yes' or 'neutral'.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    last_logits = out.logits[0, -1, :].float().log_softmax(-1)

    scores = []
    for choice in choices:
        # Leading space matches typical Llama tokenization after a newline+"Answer:".
        ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        if not ids:
            scores.append(float("-inf"))
        else:
            scores.append(last_logits[ids[0]].item())

    if priors is not None:
        scores = [s - p for s, p in zip(scores, priors)]
    return scores


@torch.no_grad()
def compute_priors(model, tokenizer, choices: list[str], prefix: str = "",
                   mc_format: bool = False, yn_only: bool = False) -> list[float]:
    """Compute unconditional log-probs for each choice using a content-free prompt.

    If few-shot prefix is provided, it's prepended so the prior is estimated in
    the same context as the actual prompts.
    """
    if yn_only:
        null = YN_NULL_PROMPT
    elif mc_format:
        null = MC_NULL_PROMPT
    else:
        null = NULL_PROMPT
    return score_choices(model, tokenizer, prefix + null, choices, priors=None)


def parse_label(text: str) -> str:
    """Parse yes/no/neutral from a generated response.

    Checks neutral/neither before no to avoid 'no' matching inside those words.
    Returns 'unparseable' when none of the expected labels are found — these are
    counted as wrong rather than silently assigned to any class.
    """
    t = text.strip().lower()
    if re.search(r'\b(neutral|neither)\b', t):
        return "neutral"
    if re.search(r'\byes\b', t):
        return "yes"
    if re.search(r'\bno\b', t):
        return "no"
    return "unparseable"


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str) -> str:
    """Greedy-decode up to 10 new tokens and parse the label from the output."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_label(text)


def gold_label(example: dict) -> str:
    """Read NormAd's "Gold Label" field; normalize stale variants from old releases."""
    for key in ("Gold Label", "gold_label", "label", "answer", "normative"):
        if key in example and example[key] is not None:
            v = str(example[key]).strip().lower()
            if v in CHOICES:
                return v
            if v in {"acceptable", "true", "1"}:
                return "yes"
            if v in {"unacceptable", "false", "0"}:
                return "no"
            if v in {"neither", "unclear"}:
                return "neutral"
    raise KeyError(f"could not find gold label in: {list(example)}")


def country(example: dict) -> str:
    """Return the canonical country (e.g. 'US', 'South_Korea'); falls back to raw value."""
    for key in ("Country", "country", "culture"):
        if key in example and example[key]:
            raw = str(example[key]).strip()
            return NORMAD_COUNTRY_MAP.get(raw.lower(), raw)
    return "Unknown"


def scenario_text(example: dict) -> str:
    for key in ("Story", "story", "scenario", "situation", "text"):
        if key in example and example[key]:
            return str(example[key]).strip()
    return ""


def evaluate_one(condition_name: str, data_path: Path, out_path: Path,
                 model_size: str = "3b", precision: str = "matched_bf16",
                 calibrate: bool = False, few_shot: int = 0, mc_format: bool = False,
                 generate: bool = False, yn_only: bool = False):
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)
    ds = load_normad(data_path)

    if yn_only:
        choices = YN_CHOICES
        template = YN_PROMPT_TEMPLATE
    elif mc_format:
        choices = MC_CHOICES
        template = MC_PROMPT_TEMPLATE
    else:
        choices = CHOICES
        template = PROMPT_TEMPLATE

    # Always exclude holdout countries so 0-shot and few-shot eval on identical examples.
    holdout_excluded = {i for i, ex in enumerate(ds) if country(ex) in HOLDOUT_COUNTRIES}
    print(f"Holdout countries excluded from eval: {sorted(HOLDOUT_COUNTRIES)} ({len(holdout_excluded)} examples)")

    if few_shot > 0:
        prefix, excluded = build_fewshot_prefix(ds, few_shot, mc_format=mc_format, yn_only=yn_only)
        excluded |= holdout_excluded
        print(f"Few-shot: 1 yes + 1 no from holdout pool, {len(excluded)} total excluded from eval")
    else:
        prefix, excluded = "", holdout_excluded

    priors = compute_priors(model, tokenizer, choices, prefix=prefix, mc_format=mc_format, yn_only=yn_only) if calibrate else None
    if calibrate:
        labels = ["A", "B", "C"] if mc_format else ["yes", "no", "neutral"]
        print("Calibration priors — " + "  ".join(f"{l}: {p:.3f}" for l, p in zip(labels, priors)))

    correct = defaultdict(int)
    total = defaultdict(int)
    n_unparseable = 0
    predictions = []

    for i, ex in enumerate(tqdm(ds, desc=f"normad/{condition_name}")):
        if i in excluded:
            continue
        if yn_only and gold_label(ex) == "neutral":
            continue
        prompt = prefix + template.format(country=country(ex), scenario=scenario_text(ex))
        if generate:
            pred = generate_answer(model, tokenizer, prompt)
            if pred == "unparseable":
                n_unparseable += 1
        else:
            scores = score_choices(model, tokenizer, prompt, choices, priors=priors)
            pred_token = choices[max(range(len(choices)), key=scores.__getitem__)]
            pred = MC_CHOICE_MAP[pred_token] if mc_format else pred_token
        gold = gold_label(ex)
        c = country(ex)
        group = culture_group(c)
        total[("all", "all")] += 1
        total[("country", c)] += 1
        total[("group", group)] += 1
        if pred == gold:
            correct[("all", "all")] += 1
            correct[("country", c)] += 1
            correct[("group", group)] += 1
        predictions.append({"country": c, "group": group, "gold": gold, "pred": pred})

    def acc(key):
        return correct[key] / total[key] if total[key] else None

    if generate and n_unparseable > 0:
        print(f"Unparseable responses: {n_unparseable} / {total[('all', 'all')]} "
              f"({100*n_unparseable/total[('all','all')]:.1f}%) — counted as wrong")

    result = {
        "condition": condition_name,
        "benchmark": "NormAd",
        "n": total[("all", "all")],
        "n_unparseable": n_unparseable if generate else 0,
        "accuracy_overall": acc(("all", "all")),
        "accuracy_by_group": {
            "Western": acc(("group", "Western")),
            "Non-Western": acc(("group", "Non-Western")),
            "Other": acc(("group", "Other")),
        },
        "accuracy_by_country": {
            c: acc(("country", c)) for (_, c) in total if _ == "country"
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
        "--model-size",
        default="3b",
        choices=["3b", "8b", "8b_instruct", "gemma4", "qwen35"],
        help="Which base model to load. '3b' = Llama-3.2-3B (default), "
             "'8b' = Llama-3.1-8B, 'gemma4' = Gemma 4 12B. Outputs for "
             "non-3B sizes are written with a size suffix (e.g. "
             "normad_sft_8b.json, normad_sft_gemma4.json) so 3B artifacts "
             "are not overwritten.",
    )
    parser.add_argument(
        "--precision",
        default="matched_bf16",
        choices=["matched_bf16", "qlora_4bit"],
        help="'matched_bf16' (default): all 4 conditions in bf16 → eliminates "
             "cross-condition precision confound. 'qlora_4bit': C1/C2/C3 in "
             "4-bit and C4 in bf16 (legacy regime; kept for back-compat).",
    )
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "NormAd"))
    parser.add_argument(
        "--out-path",
        default=None,
        help="Defaults to outputs/behavioral/normad_{condition}{size_suffix}[_calibrated].json",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Apply contextual calibration (Zhao et al. 2021): subtract per-label "
             "log-probs from a content-free prompt before argmax. Output filename "
             "gains a _calibrated suffix.",
    )
    parser.add_argument(
        "--few-shot", type=int, default=0, metavar="N",
        help="Prepend N balanced few-shot examples to each prompt. Examples are "
             "sampled from the dataset (seed=42) and excluded from eval. Output "
             "filename gains a _fsN suffix. 0 = standard 0-shot (default).",
    )
    parser.add_argument(
        "--yn-only", action="store_true",
        help="Evaluate on yes/no gold examples only (skip neutral). Uses a binary "
             "yes/no prompt so the model is not offered neutral as an option. "
             "Output gains a _yn suffix.",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Use greedy generation + label parsing instead of log-prob scoring. "
             "The model generates up to 10 tokens; 'yes'/'no'/'neutral'/'neither' "
             "are matched with word-boundary regex. Unparseable outputs are counted "
             "as wrong (not silently assigned to any class). Output gains a _gen suffix. "
             "Compatible with --few-shot; mutually exclusive with --calibrate and --mc-format.",
    )
    parser.add_argument(
        "--mc-format", action="store_true",
        help="Score letter tokens (A/B/C) instead of words (yes/no/neutral). "
             "Avoids surface-form competition (Holtzman et al. 2021, arXiv 2104.08315) "
             "where 'neutral' is structurally underscored because 'n'-initial tokens "
             "are high-frequency. Output filename gains a _mc suffix.",
    )
    args = parser.parse_args()

    if args.generate and (args.calibrate or args.mc_format):
        import sys; sys.exit("--generate is incompatible with --calibrate and --mc-format")
    if args.yn_only and args.mc_format:
        import sys; sys.exit("--yn-only is incompatible with --mc-format")

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    fs_sfx = f"_fs{args.few_shot}" if args.few_shot > 0 else ""
    yn_sfx = "_yn" if args.yn_only else ""
    gen_sfx = "_gen" if args.generate else ""
    mc_sfx = "_mc" if args.mc_format else ""
    cal_sfx = "_calibrated" if args.calibrate else ""
    out = Path(args.out_path) if args.out_path else (
        PROJECT_ROOT / "outputs" / "behavioral"
        / f"normad_{args.condition}{size_sfx}{fs_sfx}{yn_sfx}{gen_sfx}{mc_sfx}{cal_sfx}.json"
    )
    evaluate_one(
        args.condition,
        Path(args.data_path),
        out,
        model_size=args.model_size,
        precision=args.precision,
        calibrate=args.calibrate,
        few_shot=args.few_shot,
        mc_format=args.mc_format,
        generate=args.generate,
        yn_only=args.yn_only,
    )


if __name__ == "__main__":
    main()
