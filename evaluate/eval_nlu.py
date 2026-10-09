"""General NLU evaluation across alignment conditions.

Evaluates on BoolQ, CommonsenseQA (csqa), QNLI, and MRPC using log-prob
scoring after "Answer:". Supports zero-shot and neutral-fewshot modes.

Outputs JSON to outputs/behavioral/nlu_{dataset}_{condition}{suffixes}.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluate._common import (
    PROJECT_ROOT,
    build_chat_prompt,
    NEUTRAL_CHAT_SYSTEM,
    is_instruct,
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
# Extra pools for --neutral-shots N, which matches the prefix's label ratio to
# the dataset's gold ratio instead of using 1 yes / 1 no.
#
# A balanced prefix tells the model to expect 50% yes. qnli's gold really is
# ~49.5% yes, so its 1:1 prefix is already matched and nfs lifts it from chance
# to ~72%. mrpc's gold is 68.4% yes, so 1:1 under-signals yes by 18 points —
# which is what drove tulu3_dpo from +3.4 skew at 0-shot to -30.6 with nfs,
# costing it 12 accuracy points. boolq (-12.2 mismatch) and csqa (uniform shots
# against uniform gold, no skew at all) are left alone deliberately.
NEUTRAL_POOLS = {
    "qnli": {
        "yes": [
            {"question": "What is the capital of France?",
             "sentence": "Paris is the capital and most populous city of France."},
            {"question": "How many sides does a triangle have?",
             "sentence": "A triangle is a polygon with three edges and three vertices."},
            {"question": "What gas do plants absorb during photosynthesis?",
             "sentence": "During photosynthesis, plants absorb carbon dioxide from the air."},
        ],
        "no": [
            {"question": "What is the population of the Moon?",
             "sentence": "The Moon orbits the Earth at an average distance of 384,400 km."},
            {"question": "Who composed the Fifth Symphony?",
             "sentence": "The violin is a wooden string instrument played with a bow."},
            {"question": "When was the telephone invented?",
             "sentence": "Copper is a soft metal with high thermal conductivity."},
        ],
    },
    "mrpc": {
        "yes": [
            {"sentence1": "The dog ran quickly through the park.",
             "sentence2": "The canine sprinted rapidly across the park."},
            {"sentence1": "The meeting was postponed until next Tuesday.",
             "sentence2": "The meeting has been delayed to Tuesday of next week."},
            {"sentence1": "Sales increased by nearly a third last quarter.",
             "sentence2": "Last quarter saw sales rise by almost 30 percent."},
            {"sentence1": "The bridge was closed for repairs on Monday.",
             "sentence2": "On Monday, the bridge shut down so repairs could be made."},
        ],
        "no": [
            {"sentence1": "She enjoys reading books in the evening.",
             "sentence2": "He prefers watching movies at night."},
            {"sentence1": "The company opened a new office in Berlin.",
             "sentence2": "The company reported lower profits this year."},
            {"sentence1": "Heavy rain delayed the start of the match.",
             "sentence2": "The stadium holds just over forty thousand people."},
        ],
    },
}

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


def build_neutral_prefix(dataset: str, shots=None) -> str:
    if shots is None:
        shots = DATASET_CONFIGS[dataset]["neutral_shots"]
    parts = []
    for shot in shots:
        prompt = format_neutral_prompt(dataset, shot)
        parts.append(prompt + f" {shot['gold']}\n\n")
    return "".join(parts)


def build_matched_neutral_shots(dataset: str, n_shots: int, gold_yes_rate: float,
                                seed: int = 7):
    """Pick n_shots neutral shots whose label ratio matches the dataset's gold ratio.

    Only qnli and mrpc have pools for this; see NEUTRAL_POOLS for why the other
    two are excluded. n_shots=2 falls back to the hardcoded pair so existing _nfs
    results stay comparable.
    """
    if n_shots == 2:
        return list(DATASET_CONFIGS[dataset]["neutral_shots"])
    if dataset not in NEUTRAL_POOLS:
        raise ValueError(
            f"--neutral-shots {n_shots} is only supported for "
            f"{sorted(NEUTRAL_POOLS)}; {dataset} has no pool "
            f"(its 1:1 prefix is already close enough to gold)"
        )

    pool = NEUTRAL_POOLS[dataset]
    n_yes = min(max(round(n_shots * gold_yes_rate), 1), n_shots - 1)
    n_no = n_shots - n_yes
    if n_yes > len(pool["yes"]) or n_no > len(pool["no"]):
        raise ValueError(
            f"{dataset} n_shots={n_shots} needs {n_yes} yes / {n_no} no, pool holds "
            f"{len(pool['yes'])} / {len(pool['no'])}"
        )

    picked = ([dict(s, gold="yes") for s in pool["yes"][:n_yes]]
              + [dict(s, gold="no") for s in pool["no"][:n_no]])
    random.Random(seed).shuffle(picked)   # don't park the minority label at a fixed slot
    return picked


def build_real_fewshot_prefix(dataset: str, n_shots: int, seed: int = 7):
    """Build an n_shots prefix from the dataset's own train split.

    The hand-written neutral shots teach a decision boundary the benchmark may not
    share — MRPC's are near word-identical paraphrases, far cleaner than its real
    positives, and adding them moves tulu3_dpo from -3.4 skew to -31 regardless of
    their label ratio. Real demonstrations carry the dataset's own boundary.

    Train and validation are disjoint GLUE splits, so nothing is held out and no
    evaluated example is leaked. Labels are sampled to match the train split's
    natural rate, which already tracks validation's (qnli 50.0 vs 49.5% yes,
    mrpc 67.4 vs 68.4%).

    Returns (prefix, turns) where turns is the chat-format list for instruct models.
    """
    cfg = DATASET_CONFIGS[dataset]
    if cfg["hf_name"]:
        train = load_dataset(cfg["hf_path"], cfg["hf_name"], split="train")
    else:
        train = load_dataset(cfg["hf_path"], split="train")

    choices = cfg["choices"]
    rng = random.Random(seed)
    # sample a window rather than scanning 100k rows
    idx = rng.sample(range(len(train)), min(len(train), 2000))
    by_label: dict[str, list[dict]] = {c: [] for c in choices}
    for i in idx:
        ex = train[i]
        g = gold_label(dataset, ex)
        if g in by_label:
            by_label[g].append(ex)

    if set(choices) == {"yes", "no"}:
        n_tr = sum(len(v) for v in by_label.values())
        yes_rate = len(by_label["yes"]) / n_tr
        n_yes = min(max(round(n_shots * yes_rate), 1), n_shots - 1)
        want = {"yes": n_yes, "no": n_shots - n_yes}
    else:
        # round-robin the labels so every option is demonstrated
        want = {c: 0 for c in choices}
        for k in range(n_shots):
            want[choices[k % len(choices)]] += 1

    picked = []
    for lab, k in want.items():
        if len(by_label[lab]) < k:
            raise ValueError(
                f"{dataset} train sample has {len(by_label[lab])} '{lab}' rows, need {k}"
            )
        picked.extend((by_label[lab][j], lab) for j in range(k))
    rng.shuffle(picked)

    turns = [(format_prompt(dataset, ex), lab) for ex, lab in picked]
    prefix = "".join(f"{q} {a}\n\n" for q, a in turns)
    return prefix, turns


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str, choices: list[str],
                  leading_space: bool = True) -> list[float]:
    """Score each choice as the log-prob of ' {choice}' (or '{choice}') after the prompt."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc)
    last_logits = out.logits[0, -1, :].float().log_softmax(-1)

    scores = []
    for choice in choices:
        token_str = (" " + choice) if leading_space else choice
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        scores.append(last_logits[ids[0]].item() if ids else float("-inf"))
    return scores


def evaluate_one(condition_name: str, dataset: str, out_path: Path,
                 model_size: str = "3b", precision: str = "matched_bf16",
                 neutral_fewshot: bool = False, neutral_shots: int = 2,
                 few_shot: int = 0,
                 system_prompt: str | None = None):
    cfg = DATASET_CONFIGS[dataset]
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer, model = load_model_for_eval(cond, precision=precision)

    if cfg["hf_name"]:
        ds = load_dataset(cfg["hf_path"], cfg["hf_name"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_path"], split=cfg["split"])

    instruct = is_instruct(model_size)
    choices = cfg["choices"]

    prefix = ""
    fewshot_turns: list[tuple[str, str]] | None = None
    # Precedence: --few-shot N (real train-split shots) > --neutral-fewshot > 0-shot.
    if few_shot > 0:
        prefix, turns = build_real_fewshot_prefix(dataset, few_shot)
        n_yes = sum(a == "yes" for _, a in turns)
        print(f"Few-shot: {few_shot} real shots from train split"
              + (f", {n_yes} yes / {len(turns) - n_yes} no" if set(choices) == {"yes", "no"}
                 else ", round-robin over labels"))
        if instruct:
            fewshot_turns = turns
    elif neutral_fewshot:
        binary = set(choices) == {"yes", "no"}
        gold_yes_rate = 0.0
        if binary:
            golds = [g for g in (gold_label(dataset, ex) for ex in ds) if g in choices]
            gold_yes_rate = sum(g == "yes" for g in golds) / len(golds)
        shots = build_matched_neutral_shots(dataset, neutral_shots, gold_yes_rate)
        prefix = build_neutral_prefix(dataset, shots)
        if binary:
            n_yes = sum(s["gold"] == "yes" for s in shots)
            print(f"Neutral few-shot: {len(shots)} examples, {n_yes} yes / "
                  f"{len(shots) - n_yes} no (gold yes-rate {gold_yes_rate:.3f})")
        else:
            print(f"Neutral few-shot: {len(shots)} examples, one per label")
        if instruct:
            fewshot_turns = [
                (format_neutral_prompt(dataset, shot), shot["gold"])
                for shot in shots
            ]

    correct = 0
    total = 0
    predictions = []

    for ex in tqdm(ds, desc=f"nlu/{dataset}/{condition_name}"):
        gold = gold_label(dataset, ex)
        if gold not in choices:
            continue  # skip examples with missing labels (e.g. GLUE test set)

        if instruct:
            prompt = build_chat_prompt(tokenizer, format_prompt(dataset, ex), fewshot=fewshot_turns, system=system_prompt)
        else:
            prompt = prefix + format_prompt(dataset, ex)
        raw_scores = score_choices(model, tokenizer, prompt, choices, leading_space=not instruct)
        pred = choices[max(range(len(choices)), key=raw_scores.__getitem__)]

        total += 1
        if pred == gold:
            correct += 1
        predictions.append({"gold": gold, "pred": pred, "scores": list(raw_scores)})

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
                  "dpo_cult", "dpo_nocult",
                  "sft_aya_cult", "sft_aya_nocult",
                  "sftdpo_aya_cult", "sftdpo_aya_nocult"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument(
        "--model-size", default="3b",
        choices=["3b", "8b", "8b_instruct", "gemma4", "gemma4_instruct", "qwen35",
                 "olmoe", "olmoe_sft", "olmoe_instruct",
                 "gemma4_moe", "gemma4_moe_instruct",
                 "olmo3", "olmo3_sft", "olmo3_dpo"],
    )
    parser.add_argument("--precision", default="matched_bf16",
                        choices=["matched_bf16", "qlora_4bit"])
    parser.add_argument("--out-path", default=None)
    parser.add_argument(
        "--neutral-fewshot", action="store_true",
        help="Prepend culturally-agnostic few-shot examples that teach task "
             "format without domain knowledge. Output gains a _nfs suffix.",
    )
    parser.add_argument(
        "--neutral-shots", type=int, default=2, metavar="N",
        help="Number of neutral shots (default 2, the historical hardcoded set). "
             "N != 2 matches the prefix's label ratio to the dataset's gold ratio "
             "and is only supported for qnli and mrpc. Implies --neutral-fewshot. "
             "Output suffix becomes _nfsN.",
    )
    parser.add_argument(
        "--few-shot", type=int, default=0, metavar="N",
        help="Use N real demonstrations sampled from the dataset's own train split "
             "instead of the hand-written neutral shots. Train and validation are "
             "disjoint so nothing is held out. Output gains a _fsN suffix.",
    )
    parser.add_argument(
        "--system-prompt", default=None, metavar="TEXT",
        help="Explicit system message for chat-templated checkpoints. Without it\n             the template's own default is used, and Olmo-3-7B-Instruct-* defaults\n             to a function-calling system prompt the base model never sees, which\n             makes the comparison depend on an invisible variable. Pass 'neutral'\n             for 'You are a helpful assistant.'. No effect on non-instruct sizes.\n             Output gains a _sys suffix.",
    )
    args = parser.parse_args()
    if args.system_prompt == "neutral":
        args.system_prompt = NEUTRAL_CHAT_SYSTEM

    if args.neutral_shots != 2:
        args.neutral_fewshot = True
    if args.few_shot > 0 and args.neutral_fewshot:
        parser.error("--few-shot and --neutral-fewshot/--neutral-shots are mutually exclusive")

    size_sfx = "" if args.model_size == "3b" else f"_{args.model_size}"
    if args.few_shot > 0:
        nfs_sfx = f"_fs{args.few_shot}"
    elif not args.neutral_fewshot:
        nfs_sfx = ""
    else:
        # _nfs stays bare at the historical 2 shots so old filenames keep matching
        nfs_sfx = "_nfs" if args.neutral_shots == 2 else f"_nfs{args.neutral_shots}"
    sys_sfx = "_sys" if args.system_prompt else ""

    out = Path(args.out_path) if args.out_path else (
        PROJECT_ROOT / "outputs" / "behavioral"
        / f"nlu_{args.dataset}_{args.condition}{size_sfx}{nfs_sfx}{sys_sfx}.json"
    )
    evaluate_one(
        args.condition,
        args.dataset,
        out,
        model_size=args.model_size,
        precision=args.precision,
        neutral_fewshot=args.neutral_fewshot,
        neutral_shots=args.neutral_shots,
        few_shot=args.few_shot,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
