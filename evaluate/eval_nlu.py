"""General NLU evaluation across alignment conditions.

Evaluates on BoolQ, CommonsenseQA (csqa), QNLI, and MRPC using log-prob
scoring after "Answer:". Supports zero-shot and neutral-fewshot modes.

Outputs JSON to outputs/behavioral/nlu_{dataset}_{condition}{suffixes}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    load_model_for_eval,
    resolve_condition,
)

# ---------------------------------------------------------------------------
# Prompt templates — all end with "Answer:" for consistent leading-space scoring
# ---------------------------------------------------------------------------

BOOLQ_PROMPT = (
    "Passage: {passage}\n"
    "Question: {question}\n"
    "Answer:"
)

CSQA_PROMPT = (
    "Question: {question}\n"
    "A. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nE. {option_e}\n"
    "Answer:"
)

QNLI_PROMPT = (
    "Does the following context answer the question?\n"
    "Question: {question}\n"
    "Context: {sentence}\n"
    "Answer yes if the context answers the question, no if it does not.\n"
    "Answer:"
)

MRPC_PROMPT = (
    "Are the following two sentences paraphrases of each other?\n"
    "Sentence 1: {sentence1}\n"
    "Sentence 2: {sentence2}\n"
    "Answer yes if they are paraphrases, no if they are not.\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# Neutral few-shot shots — teach task format, not domain knowledge
# ---------------------------------------------------------------------------

BOOLQ_NEUTRAL_SHOTS = [
    # gold=yes
    {
        "passage": "Water freezes at 0 degrees Celsius at standard atmospheric pressure and becomes ice.",
        "question": "Does water turn to ice at 0 degrees Celsius?",
        "gold": "yes",
    },
    # gold=no
    {
        "passage": "The Earth takes approximately 365 days to complete one orbit around the Sun.",
        "question": "Does the Earth orbit the Moon?",
        "gold": "no",
    },
]

CSQA_NEUTRAL_SHOTS = [
    # gold=A
    {"question": "What do you use to write on paper?",
     "option_a": "pen", "option_b": "ruler", "option_c": "keyboard", "option_d": "phone", "option_e": "hammer",
     "gold": "A"},
    # gold=B
    {"question": "What color is grass?",
     "option_a": "red", "option_b": "green", "option_c": "blue", "option_d": "yellow", "option_e": "purple",
     "gold": "B"},
    # gold=C
    {"question": "How many days are in a week?",
     "option_a": "five", "option_b": "eight", "option_c": "seven", "option_d": "ten", "option_e": "three",
     "gold": "C"},
    # gold=D
    {"question": "What is 2 + 2?",
     "option_a": "three", "option_b": "five", "option_c": "six", "option_d": "four", "option_e": "ten",
     "gold": "D"},
    # gold=E
    {"question": "What do birds use to fly?",
     "option_a": "fins", "option_b": "legs", "option_c": "tails", "option_d": "ears", "option_e": "wings",
     "gold": "E"},
]

QNLI_NEUTRAL_SHOTS = [
    # gold=yes (entailment — context answers the question)
    {"question": "What is the capital of France?",
     "sentence": "Paris is the capital and most populous city of France.",
     "gold": "yes"},
    # gold=no (not entailment — context does not answer the question)
    {"question": "What is the population of the Moon?",
     "sentence": "The Moon orbits the Earth at an average distance of 384,400 km.",
     "gold": "no"},
]

MRPC_NEUTRAL_SHOTS = [
    # gold=yes (paraphrase)
    {"sentence1": "The dog ran quickly through the park.",
     "sentence2": "The canine sprinted rapidly across the park.",
     "gold": "yes"},
    # gold=no (not paraphrase)
    {"sentence1": "She enjoys reading books in the evening.",
     "sentence2": "He prefers watching movies at night.",
     "gold": "no"},
]

# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "boolq": {
        "hf_path": "google/boolq",
        "hf_name": None,
        "split": "validation",
        "choices": ["yes", "no"],
        "neutral_shots": BOOLQ_NEUTRAL_SHOTS,
    },
    "csqa": {
        "hf_path": "tau/commonsense_qa",
        "hf_name": None,
        "split": "validation",
        "choices": ["A", "B", "C", "D", "E"],
        "neutral_shots": CSQA_NEUTRAL_SHOTS,
    },
    "qnli": {
        "hf_path": "nyu-mll/glue",
        "hf_name": "qnli",
        "split": "validation",
        "choices": ["yes", "no"],
        "neutral_shots": QNLI_NEUTRAL_SHOTS,
    },
    "mrpc": {
        "hf_path": "nyu-mll/glue",
        "hf_name": "mrpc",
        "split": "validation",
        "choices": ["yes", "no"],
        "neutral_shots": MRPC_NEUTRAL_SHOTS,
    },
}


def format_prompt(dataset: str, ex: dict) -> str:
    if dataset == "boolq":
        return BOOLQ_PROMPT.format(passage=ex["passage"], question=ex["question"])
    if dataset == "csqa":
        opts = {l: c for l, c in zip(ex["choices"]["label"], ex["choices"]["text"])}
        return CSQA_PROMPT.format(
            question=ex["question"],
            option_a=opts["A"], option_b=opts["B"], option_c=opts["C"],
            option_d=opts["D"], option_e=opts["E"],
        )
    if dataset == "qnli":
        return QNLI_PROMPT.format(question=ex["question"], sentence=ex["sentence"])
    if dataset == "mrpc":
        return MRPC_PROMPT.format(sentence1=ex["sentence1"], sentence2=ex["sentence2"])
    raise ValueError(f"unknown dataset: {dataset}")


def format_neutral_prompt(dataset: str, shot: dict) -> str:
    if dataset == "boolq":
        return BOOLQ_PROMPT.format(passage=shot["passage"], question=shot["question"])
    if dataset == "csqa":
        return CSQA_PROMPT.format(
            question=shot["question"],
            option_a=shot["option_a"], option_b=shot["option_b"],
            option_c=shot["option_c"], option_d=shot["option_d"],
            option_e=shot["option_e"],
        )
    if dataset == "qnli":
        return QNLI_PROMPT.format(question=shot["question"], sentence=shot["sentence"])
    if dataset == "mrpc":
        return MRPC_PROMPT.format(sentence1=shot["sentence1"], sentence2=shot["sentence2"])
    raise ValueError(f"unknown dataset: {dataset}")


def gold_label(dataset: str, ex: dict) -> str:
    if dataset == "boolq":
        return "yes" if ex["answer"] else "no"
    if dataset == "csqa":
        return ex["answerKey"]
    if dataset == "qnli":
        # 0 = entailment (context answers question) → yes
        # 1 = not_entailment → no
        return "yes" if ex["label"] == 0 else "no"
    if dataset == "mrpc":
        # 1 = paraphrase → yes; 0 = not paraphrase → no
        return "yes" if ex["label"] == 1 else "no"
    raise ValueError(f"unknown dataset: {dataset}")


def build_neutral_prefix(dataset: str) -> str:
    shots = DATASET_CONFIGS[dataset]["neutral_shots"]
    parts = []
    for shot in shots:
        prompt = format_neutral_prompt(dataset, shot)
        parts.append(prompt + f" {shot['gold']}\n\n")
    return "".join(parts)


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str, choices: list[str]) -> list[float]:
    """Score each choice as the log-prob of ' {choice}' after the prompt."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    last_logits = out.logits[0, -1, :].float().log_softmax(-1)

    scores = []
    for choice in choices:
        ids = tokenizer.encode(" " + choice, add_special_tokens=False)
        scores.append(last_logits[ids[0]].item() if ids else float("-inf"))
    return scores


def evaluate_one(condition_name: str, dataset: str, out_path: Path,
                 model_size: str = "3b", precision: str = "matched_bf16",
                 neutral_fewshot: bool = False):
    cfg = DATASET_CONFIGS[dataset]
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)

    if cfg["hf_name"]:
        ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_path"], split=cfg["split"])

    choices = cfg["choices"]
    prefix = build_neutral_prefix(dataset) if neutral_fewshot else ""
    if neutral_fewshot:
        print(f"Neutral few-shot: {len(cfg['neutral_shots'])} examples")

    correct = 0
    total = 0
    predictions = []

    for ex in tqdm(ds, desc=f"nlu/{dataset}/{condition_name}"):
        gold = gold_label(dataset, ex)
        if gold not in choices:
            continue  # skip examples with missing labels (e.g. GLUE test set)

        prompt = prefix + format_prompt(dataset, ex)
        scores = score_choices(model, tokenizer, prompt, choices)
        pred = choices[max(range(len(choices)), key=scores.__getitem__)]

        total += 1
        if pred == gold:
            correct += 1
        predictions.append({"gold": gold, "pred": pred})

    accuracy = correct / total if total else None
    result = {
        "condition": condition_name,
        "benchmark": dataset,
        "n": total,
        "accuracy_overall": accuracy,
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}: overall={accuracy:.3f}")


ALL_CONDITIONS = ["base", "dpo", "sft", "sftdpo",
                  "dpo_coig", "dpo_pku", "sftdpo_coig", "sftdpo_pku",
                  "tulu3_sft", "tulu3_dpo",
                  "sft_aya_cult", "sft_aya_nocult",
                  "sftdpo_aya_cult", "sftdpo_aya_nocult"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument(
        "--model-size", default="3b",
        choices=["3b", "8b", "8b_instruct", "gemma4", "qwen35"],
    )
    parser.add_argument("--precision", default="matched_bf16",
                        choices=["matched_bf16", "qlora_4bit"])
    parser.add_argument("--out-path", default=None)
    parser.add_argument(
        "--neutral-fewshot", action="store_true",
        help="Prepend culturally-agnostic few-shot examples that teach task "
             "format without domain knowledge. Output gains a _nfs suffix.",
    )
    args = parser.parse_args()

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    nfs_sfx = "_nfs" if args.neutral_fewshot else ""

    out = Path(args.out_path) if args.out_path else (
        PROJECT_ROOT / "outputs" / "behavioral"
        / f"nlu_{args.dataset}_{args.condition}{size_sfx}{nfs_sfx}.json"
    )
    evaluate_one(
        args.condition,
        args.dataset,
        out,
        model_size=args.model_size,
        precision=args.precision,
        neutral_fewshot=args.neutral_fewshot,
    )


if __name__ == "__main__":
    main()
