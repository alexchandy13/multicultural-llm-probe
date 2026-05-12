"""CARE preference-alignment evaluation, mirroring the NormAd evaluator's structure.

CARE pairs culturally-appropriate vs. inappropriate responses per scenario. We compute
the log-prob of each side under the model and report win rate by culture group.
"""
from __future__ import annotations

import argparse
import json
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


PROMPT_TEMPLATE = (
    "Country: {country}\n"
    "Question: {question}\n"
    "Response:"
)


def load_care(path: Path):
    if path.exists() and any(path.iterdir()):
        return load_from_disk(str(path))
    ds = load_dataset("MichiganNLP/CARE")
    return ds["test"] if "test" in ds else ds["train"]


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    """Sum log-prob of continuation tokens conditional on prompt."""
    device = next(model.parameters()).device
    full = prompt + " " + continuation
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(device)
    out = model(full_ids)
    logits = out.logits[0, :-1, :].float().log_softmax(-1)
    targets = full_ids[0, 1:]

    n_prompt = prompt_ids.shape[1]
    cont_mask = torch.zeros_like(targets, dtype=torch.bool)
    cont_mask[n_prompt - 1:] = True

    token_lp = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp[cont_mask].sum().item()


def country(example: dict) -> str:
    for key in ("country", "Country", "culture"):
        if key in example and example[key]:
            return str(example[key]).strip()
    return "Unknown"


def question(example: dict) -> str:
    for key in ("question", "scenario", "prompt"):
        if key in example and example[key]:
            return str(example[key])
    return ""


def chosen_rejected(example: dict) -> tuple[str, str]:
    if "chosen" in example and "rejected" in example:
        return example["chosen"], example["rejected"]
    if "preferred" in example and "dispreferred" in example:
        return example["preferred"], example["dispreferred"]
    raise KeyError(f"could not find preference pair in: {list(example)}")


def evaluate_one(condition_name: str, data_path: Path, out_path: Path):
    cond = resolve_condition(condition_name)
    tokenizer, model = load_model_for_eval(cond)
    ds = load_care(data_path)

    wins = defaultdict(int)
    total = defaultdict(int)
    rows = []

    for ex in tqdm(ds, desc=f"care/{condition_name}"):
        c = country(ex)
        group = culture_group(c)
        prompt = PROMPT_TEMPLATE.format(country=c, question=question(ex))
        try:
            chosen, rejected = chosen_rejected(ex)
        except KeyError:
            continue

        lp_chosen = sequence_logprob(model, tokenizer, prompt, chosen)
        lp_rejected = sequence_logprob(model, tokenizer, prompt, rejected)
        win = lp_chosen > lp_rejected

        total[("all", "all")] += 1
        total[("country", c)] += 1
        total[("group", group)] += 1
        if win:
            wins[("all", "all")] += 1
            wins[("country", c)] += 1
            wins[("group", group)] += 1
        rows.append({"country": c, "group": group, "win": win, "lp_chosen": lp_chosen, "lp_rejected": lp_rejected})

    def rate(key):
        return wins[key] / total[key] if total[key] else None

    result = {
        "condition": condition_name,
        "benchmark": "CARE",
        "n": total[("all", "all")],
        "win_rate_overall": rate(("all", "all")),
        "win_rate_by_group": {
            "Western": rate(("group", "Western")),
            "Non-Western": rate(("group", "Non-Western")),
            "Other": rate(("group", "Other")),
        },
        "win_rate_by_country": {
            c: rate(("country", c)) for (_, c) in total if _ == "country"
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}: win_rate={result['win_rate_overall']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["base", "sft", "dpo", "instruct"])
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "data" / "CARE"))
    parser.add_argument("--out-path", default=None)
    args = parser.parse_args()

    out = Path(args.out_path) if args.out_path else (
        PROJECT_ROOT / "outputs" / "behavioral" / f"care_{args.condition}.json"
    )
    evaluate_one(args.condition, Path(args.data_path), out)


if __name__ == "__main__":
    main()
