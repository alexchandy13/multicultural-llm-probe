"""Extends upstream's `load_dataset_neuron_scores` with custom dataset blocks.

Adds loaders for:
  - normad_yn / normadcontrol  (yes/no NormAd + null-content baseline)
  - culturalbench / culturalbenchcontrol  (CulturalBench yes/no + null-country baseline)

We do this by monkey-patching upstream's loader before any caller imports it.
Import this module first (or set `PYTHONPATH` to include `culnig/`); the patch is
idempotent.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path(__file__).resolve().parent / "_upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

import dataset as _upstream_dataset  # noqa: E402
from datasets import load_dataset, Dataset as HFDataset  # noqa: E402



# Neutral fewshot prefix — identical to eval_normad.py's NEUTRAL_SHOTS_NORMAD.
# Prepended to every prompt so CULNIG sees the same context as behavioral eval.
_FEWSHOT_PREFIX = (
    "Is the Earth round?\nAnswer: yes\n\n"
    "Is fire cold?\nAnswer: no\n\n"
)

# Binary yes/no prompts — exact match to eval_normad.py's YN_WORD_PROMPTS,
# with fewshot prefix prepended. Answer token is "yes" or "no".
NORMAD_YN_PROMPTS = [
    _FEWSHOT_PREFIX + (
        "You will be given a short story involving 2 or more characters about everyday "
        "situations. You will also be given a country of relevance. Read the story and "
        "determine if the character's behavior aligns with the social norms of that country.\n"
        "Respond **only** with yes or no.\n\n"
        "country: {country}\nStory: {story}\nAnswer:"
    ),
    _FEWSHOT_PREFIX + (
        "Read a short story describing everyday events with at least two characters. "
        "You'll also receive a country. Your task is to judge whether the character's "
        "actions are in line with the cultural expectations of that country.\n"
        "Answer strictly with yes or no.\n\n"
        "country: {country}\nStory: {story}\nAnswer:"
    ),
    _FEWSHOT_PREFIX + (
        "A short scenario featuring multiple characters will be provided, along with the "
        "cultural background of a specific country. Your role is to determine how well "
        "the behavior shown fits that country's social norms.\n"
        "Only respond with yes or no.\n\n"
        "country: {country}\nStory: {story}\nAnswer:"
    ),
    _FEWSHOT_PREFIX + (
        "Given a short daily-life story involving multiple characters, along with the "
        "country, assess whether the behavior depicted fits within the social norms of "
        "that culture.\nAnswer with just yes or no.\n\n"
        "country: {country}\nStory: {story}\nAnswer:"
    ),
]

# Countries held out from evaluation and used as the few-shot pool in eval_normad.py.
# Stored in NormAd's raw format (lowercase + underscore) to match the 'Country' field.
HOLDOUT_NORMAD_COUNTRIES = {
    "syria", "indonesia", "colombia", "austria", "north_macedonia", "sweden",
}


_original_loader = _upstream_dataset.load_dataset_neuron_scores
_patched = False


def _normad_yn_block(tokenizer, target_data):
    """Build the processed dataset for `normad_yn`.

    Like upstream's normad block but:
      - Filters out neutral-gold examples entirely.
      - Excludes HOLDOUT_NORMAD_COUNTRIES (same countries held out in eval_normad.py).
      - Uses a binary prompt (answer 1 or 2, no neutral option).
      - Shuffles yes/no -> 1/2 mapping per-example using a seeded hash.
    """
    c2n = _upstream_dataset.COUNTRY_TO_NAME["normad"]
    rev_c2n = {v: k for k, v in c2n.items()}

    dataset = load_dataset("akhilayerukola/NormAd", split="train")
    # Drop holdout countries and neutral-gold examples.
    dataset = dataset.filter(
        lambda x: x["Country"] not in HOLDOUT_NORMAD_COUNTRIES
        and x["Gold Label"] in ("yes", "no")
    )

    all_countries = np.unique(dataset["Country"]).tolist()

    if target_data != "all":
        ids = []
        for country in all_countries:
            cdata = dataset.filter(lambda x, c=country: x["Country"] == c)
            for label in ["yes", "no"]:
                ldata = cdata.filter(lambda x, l=label: x["Gold Label"] == l)
                n = len(ldata)
                half = n // 2
                if target_data == "neuron":
                    ids.extend(ldata.select(range(half))["ID"])
                else:
                    ids.extend(ldata.select(range(half, n))["ID"])
        dataset = dataset.filter(lambda x: x["ID"] in ids)

    def make_preprocess(instruction, inst_idx):
        def preprocess(examples):
            gold = examples["Gold Label"]
            if gold not in ("yes", "no"):
                raise ValueError(f"Unexpected label in normad_yn: {gold}")

            country_key = rev_c2n.get(examples["Country"], examples["Country"])
            input_text = instruction.format(
                country=examples["Country"],
                story=examples["Story"],
            )
            tokenized = tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            )
            return {
                "input_text": input_text,
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "label": gold,
                "country": country_key,
                "id": str(examples["ID"]),
                "instruction_id": inst_idx,
                "dataset_name": "normad_yn",
                "options": [gold],
            }
        return preprocess

    processed = []
    for inst_idx, instruction in enumerate(NORMAD_YN_PROMPTS):
        processed.append(
            dataset.map(
                make_preprocess(instruction, inst_idx),
                remove_columns=dataset.column_names,
                num_proc=1,
            )
        )
    return processed


def _normadcontrol_block(tokenizer, target_countries, target_data):
    """Build the processed dataset for `normadcontrol`.

    Same format as normad_yn (yes/no labels, same 4 prompt templates, fewshot
    prefix) but with all cultural content removed:
      - story="" — no story text
      - country="" — no country injected
    This makes the only variable between normad_yn and normadcontrol the
    presence of cultural content, not prompt format, so gradient comparisons
    are not confounded by format differences.
    """
    c2n = _upstream_dataset.COUNTRY_TO_NAME["normad"]
    rev_c2n = {v: k for k, v in c2n.items()}

    dataset = load_dataset("akhilayerukola/NormAd", split="train")
    # Same filters as normad_yn: drop holdout countries and neutral labels.
    dataset = dataset.filter(
        lambda x: x["Country"] not in HOLDOUT_NORMAD_COUNTRIES
        and x["Gold Label"] in ("yes", "no")
    )

    all_countries = np.unique(dataset["Country"]).tolist()

    if target_countries is not None:
        target = [c2n[c] for c in target_countries]
        dataset = dataset.filter(lambda x: x["Country"] in target)
    else:
        target = all_countries

    if target_data != "all":
        ids = []
        for country in target:
            cdata = dataset.filter(lambda x, c=country: x["Country"] == c)
            for label in ["yes", "no"]:
                ldata = cdata.filter(lambda x, l=label: x["Gold Label"] == l)
                n = len(ldata)
                half = n // 2
                if target_data == "neuron":
                    ids.extend(ldata.select(range(half))["ID"])
                else:
                    ids.extend(ldata.select(range(half, n))["ID"])
        dataset = dataset.filter(lambda x: x["ID"] in ids)

    def make_preprocess(instruction, inst_idx):
        def preprocess(examples):
            gold = examples["Gold Label"]
            if gold not in ("yes", "no"):
                raise ValueError(f"Unexpected label in normadcontrol: {gold}")

            # Inject empty country and story — no cultural content.
            input_text = instruction.format(country="", story="")
            tokenized = tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            )
            return {
                "input_text": input_text,
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "label": gold,
                "country": rev_c2n.get(examples["Country"], examples["Country"]),
                "id": str(examples["ID"]),
                "instruction_id": inst_idx,
                "dataset_name": "normadcontrol",
                "options": [gold],
            }
        return preprocess

    processed = []
    for inst_idx, instruction in enumerate(NORMAD_YN_PROMPTS):
        processed.append(
            dataset.map(
                make_preprocess(instruction, inst_idx),
                remove_columns=dataset.column_names,
                num_proc=1,
            )
        )
    return processed


# ── CulturalBench ────────────────────────────────────────────────────────────

CULTURALBENCH_DATA_PATH = PROJECT_ROOT / "data" / "culturalbench_reformatted.json"

# Countries that take "the" as a determiner — mirrors eval_culturalbench.py.
_CB_THE_COUNTRIES = {
    "Netherlands", "United Kingdom", "United States",
    "United Arab Emirates", "Dominican Republic", "Philippines",
    "Czech Republic",
}

_CB_PROMPT_SUFFIX = "\nAnswer with yes or no.\nAnswer:"


def _country_with_article(country: str) -> str:
    return f"the {country}" if country in _CB_THE_COUNTRIES else country


def _cb_rows(target_data: str) -> list[dict]:
    """Load and optionally half-split CulturalBench rows."""
    rows = [r for r in json.loads(CULTURALBENCH_DATA_PATH.read_text())
            if r.get("reformatted_prompt")]
    if target_data == "all":
        return rows
    groups: dict = defaultdict(list)
    for i, r in enumerate(rows):
        label = "yes" if r["answer"] else "no"
        groups[(r["country"], label)].append(i)
    selected: set = set()
    for indices in groups.values():
        half = len(indices) // 2
        selected.update(indices[:half] if target_data == "neuron" else indices[half:])
    return [r for i, r in enumerate(rows) if i in selected]


def _culturalbench_block(tokenizer, target_data: str) -> list:
    """CulturalBench yes/no scoring dataset, matching eval_culturalbench.py format."""
    rows = _cb_rows(target_data)

    def make_example(r):
        gold = "yes" if r["answer"] else "no"
        prompt = _FEWSHOT_PREFIX + r["reformatted_prompt"] + _CB_PROMPT_SUFFIX
        tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {
            "input_text": prompt,
            "input_ids": tok["input_ids"][0].tolist(),
            "attention_mask": tok["attention_mask"][0].tolist(),
            "label": gold,
            "country": r["country"],
            "id": str(r["data_idx"]),
            "instruction_id": 0,
            "dataset_name": "culturalbench",
            "options": [gold],
        }

    return [HFDataset.from_list([make_example(r) for r in rows])]


def _culturalbenchcontrol_block(tokenizer, target_data: str) -> list:
    """CulturalBench control: same prompts but country name stripped.

    Replaces 'In {country with article},' with 'In ,' so the only difference
    from culturalbench is the presence of a named country.
    """
    rows = _cb_rows(target_data)

    def make_example(r):
        gold = "yes" if r["answer"] else "no"
        src = f"In {_country_with_article(r['country'])},"
        controlled = r["reformatted_prompt"].replace(src, "In ,", 1)
        prompt = _FEWSHOT_PREFIX + controlled + _CB_PROMPT_SUFFIX
        tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {
            "input_text": prompt,
            "input_ids": tok["input_ids"][0].tolist(),
            "attention_mask": tok["attention_mask"][0].tolist(),
            "label": gold,
            "country": r["country"],
            "id": str(r["data_idx"]),
            "instruction_id": 0,
            "dataset_name": "culturalbenchcontrol",
            "options": [gold],
        }

    return [HFDataset.from_list([make_example(r) for r in rows])]


def _patched_loader(dataset_names, tokenizer, batch_size,
                    target_countries=None, target_data="all"):
    """Drop-in replacement: handle `normadcontrol` and `normad_yn`, delegate the rest."""
    from datasets import concatenate_datasets
    import torch
    from torch.nn.utils.rnn import pad_sequence

    _local = {"normadcontrol", "normad_yn", "culturalbench", "culturalbenchcontrol"}
    has_ctrl = "normadcontrol" in dataset_names
    has_yn = "normad_yn" in dataset_names
    has_cb = "culturalbench" in dataset_names
    has_cb_ctrl = "culturalbenchcontrol" in dataset_names
    remaining = [d for d in dataset_names if d not in _local]
    extra_processed = []
    if has_ctrl:
        extra_processed += _normadcontrol_block(tokenizer, target_countries, target_data)
    if has_yn:
        extra_processed += _normad_yn_block(tokenizer, target_data)
    if has_cb:
        extra_processed += _culturalbench_block(tokenizer, target_data)
    if has_cb_ctrl:
        extra_processed += _culturalbenchcontrol_block(tokenizer, target_data)

    if remaining:
        # Delegate the rest to the unmodified upstream loader.
        base_loader = _original_loader(
            remaining, tokenizer, batch_size,
            target_countries=target_countries, target_data=target_data,
        )
        base_dataset = base_loader.dataset
        combined = concatenate_datasets([base_dataset] + extra_processed) if extra_processed else base_dataset
    else:
        combined = concatenate_datasets(extra_processed) if extra_processed else None
        if combined is None:
            raise ValueError("No datasets specified")

    def collator(batch):
        input_texts = [item["input_text"] for item in batch]
        input_ids = pad_sequence(
            [torch.tensor(item["input_ids"]) for item in batch],
            batch_first=True, padding_value=tokenizer.pad_token_id, padding_side="left",
        )
        attention_mask = pad_sequence(
            [torch.tensor(item["attention_mask"]) for item in batch],
            batch_first=True, padding_value=0, padding_side="left",
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": [item["label"] for item in batch],
            "countries": [item["country"] for item in batch],
            "input_texts": input_texts,
            "ids": [item["id"] for item in batch],
            "instruction_ids": [item["instruction_id"] for item in batch],
            "dataset_names": [item["dataset_name"] for item in batch],
            "options": [item["options"] for item in batch],
        }

    return torch.utils.data.DataLoader(
        combined, batch_size=batch_size, collate_fn=collator,
        shuffle=False, pin_memory=True,
    )


def install():
    """Idempotently swap the upstream loader for our patched version."""
    global _patched
    if _patched:
        return
    _upstream_dataset.load_dataset_neuron_scores = _patched_loader
    _patched = True


install()
