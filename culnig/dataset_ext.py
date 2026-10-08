"""Extends upstream's `load_dataset_neuron_scores` with custom dataset blocks.

Adds loaders for:
  - normad_yn / normadcontrol  (yes/no NormAd + null-content baseline)
  - culturalbench / culturalbenchcontrol  (CulturalBench yes/no + null-country baseline)
  - blend / blendcontrol  (BLEnD MCQ + question-stripped baseline, both with NFS prefix)

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


# ── BLEnD ────────────────────────────────────────────────────────────────────

BLEND_DATA_PATH = PROJECT_ROOT / "data" / "BLEnD"

_BLEND_INST = (
    " Without any explanation, choose only one from the given alphabet choices"
    '(e.g., A, B, C). Provide as JSON format: {"answer_choice":""}'
)
_BLEND_SCORING_SUFFIX = '{"answer_choice":"'

# Same shots as eval_blend.py NEUTRAL_SHOTS_BLEND: 4 culturally-agnostic MCQ
# examples, one per answer letter. Raw text for CULNIG (stripped chat template).
_NEUTRAL_SHOTS_BLEND = [
    ("Which planet is farthest from the Sun in our solar system?" + _BLEND_INST +
     "\n\nA. Neptune\nB. Jupiter\nC. Saturn\nD. Uranus\n\nAnswer:", "A"),
    ("What is the chemical symbol for water?" + _BLEND_INST +
     "\n\nA. O2\nB. H2O\nC. CO2\nD. NaCl\n\nAnswer:", "B"),
    ("How many sides does a hexagon have?" + _BLEND_INST +
     "\n\nA. Five\nB. Seven\nC. Six\nD. Eight\n\nAnswer:", "C"),
    ("What is the largest ocean on Earth?" + _BLEND_INST +
     "\n\nA. Atlantic\nB. Indian\nC. Arctic\nD. Pacific\n\nAnswer:", "D"),
]

# NFS prefix as raw text: each shot ends with the scoring suffix + answer letter.
_FEWSHOT_PREFIX_BLEND = "".join(
    q + _BLEND_SCORING_SUFFIX + lbl + '"}\n\n'
    for q, lbl in _NEUTRAL_SHOTS_BLEND
)

# blendcontrol prompt template — question text stripped, only options remain.
# Mirrors upstream CULNIG's blendcontrol design.
_BLEND_CONTROL_TMPL = (
    "Without any explanation, choose only one from the given alphabet choices"
    '(e.g., A, B, C). Provide as JSON format: {{"answer_choice":""}}'
    "\n\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\n\nAnswer:"
)


def _load_blend_dedup():
    """Load BLEnD test split, deduplicating to max 5 MCQ variants per (country, ID)."""
    has_data = (BLEND_DATA_PATH.exists()
                and any(p for p in BLEND_DATA_PATH.iterdir() if p.name != ".gitkeep"))
    if has_data:
        from datasets import load_from_disk
        ds = load_from_disk(str(BLEND_DATA_PATH))
    else:
        ds = load_dataset("nayeon212/BLEnD", "multiple-choice-questions", split="test")
    country_id_mcqids: dict = {}
    for item in ds:
        key = (item["country"], item["ID"])
        if key not in country_id_mcqids:
            country_id_mcqids[key] = []
        if len(country_id_mcqids[key]) < 5:
            country_id_mcqids[key].append(item["MCQID"])
    valid_mcqids = {m for ids in country_id_mcqids.values() for m in ids}
    return ds.filter(lambda x: x["MCQID"] in valid_mcqids)


def _blend_split(ds, target_data: str):
    """Split BLEnD into neuron/non_neuron halves by (country, answer_idx) groups."""
    if target_data == "all":
        return ds
    groups: dict = defaultdict(list)
    for item in ds:
        groups[(item["country"], item["answer_idx"])].append(item["MCQID"])
    selected: set = set()
    for mcqids in groups.values():
        half = len(mcqids) // 2
        selected.update(mcqids[:half] if target_data == "neuron" else mcqids[half:])
    return ds.filter(lambda x: x["MCQID"] in selected)


def _blend_block(tokenizer, target_data: str) -> list:
    """BLEnD MCQ with NFS prefix, label = correct answer letter (A/B/C/D)."""
    ds = _blend_split(_load_blend_dedup(), target_data)

    def make_example(item):
        gold = item["answer_idx"]
        prompt = _FEWSHOT_PREFIX_BLEND + item["prompt"] + _BLEND_SCORING_SUFFIX
        tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {
            "input_text": prompt,
            "input_ids": tok["input_ids"][0].tolist(),
            "attention_mask": tok["attention_mask"][0].tolist(),
            "label": gold,
            "country": item["country"],
            "id": str(item["MCQID"]),
            "instruction_id": 0,
            "dataset_name": "blend",
            "options": [gold],
        }

    return [HFDataset.from_list([make_example(item) for item in ds])]


def _blendcontrol_block(tokenizer, target_data: str) -> list:
    """BLEnD control: question text stripped, only answer options kept.

    Same NFS prefix and scoring suffix as blend so the only difference is the
    absence of the cultural question. Mirrors upstream CULNIG's blendcontrol.
    """
    import json as _json
    ds = _blend_split(_load_blend_dedup(), target_data)

    def make_example(item):
        gold = item["answer_idx"]
        choices = _json.loads(item["choices"])
        ctrl_text = _BLEND_CONTROL_TMPL.format(
            option_a=choices["A"], option_b=choices["B"],
            option_c=choices["C"], option_d=choices["D"],
        )
        prompt = _FEWSHOT_PREFIX_BLEND + ctrl_text + _BLEND_SCORING_SUFFIX
        tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {
            "input_text": prompt,
            "input_ids": tok["input_ids"][0].tolist(),
            "attention_mask": tok["attention_mask"][0].tolist(),
            "label": gold,
            "country": item["country"],
            "id": str(item["MCQID"]),
            "instruction_id": 0,
            "dataset_name": "blendcontrol",
            "options": [gold],
        }

    return [HFDataset.from_list([make_example(item) for item in ds])]


# ── CulturalBench ────────────────────────────────────────────────────────────

CULTURALBENCH_DATA_PATH = PROJECT_ROOT / "data" / "culturalbench_reformatted.json"

# Countries that take "the" as a determiner — mirrors eval_culturalbench.py.
_CB_THE_COUNTRIES = {
    "Netherlands", "United Kingdom", "United States",
    "United Arab Emirates", "Dominican Republic", "Philippines",
    "Czech Republic",
}

_CB_PROMPT_SUFFIX = "\nAnswer with yes or no.\nAnswer:"

# CulturalBench gets its own prefix: gold is 27% yes, so the 1 yes / 1 no pair in
# _FEWSHOT_PREFIX signals 50% and pushes predictions toward yes — the direction
# every condition already over-predicts. These are the 4 shots eval_culturalbench.py
# builds with --neutral-shots 4 (1 yes / 3 no, seed 7), so CULNIG and the behavioral
# eval see the same context. NormAd keeps _FEWSHOT_PREFIX: its gold is 51.9% yes,
# which the 1:1 pair already matches.
_CB_FEWSHOT_PREFIX = (
    "Is the moon larger than the sun?\nAnswer: no\n\n"
    "Is fire cold?\nAnswer: no\n\n"
    "Is the Earth round?\nAnswer: yes\n\n"
    "Do fish breathe through lungs?\nAnswer: no\n\n"
)


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
        prompt = _CB_FEWSHOT_PREFIX + r["reformatted_prompt"] + _CB_PROMPT_SUFFIX
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
    """CulturalBench control: the question removed, prefix and answer cue kept.

    Mirrors normadcontrol, which empties both country and story. Every control
    prompt is therefore identical; the per-item variation is only which label's
    gradient is taken.
    """
    rows = _cb_rows(target_data)

    def make_example(r):
        gold = "yes" if r["answer"] else "no"
        # Drop the question entirely, leaving the fewshot prefix and the answer cue.
        # This mirrors normadcontrol, which empties both country and story rather
        # than blanking the country alone, so the two benchmarks' controls now
        # measure the same thing: the model's response with no question content.
        #
        # The previous version blanked only the country name, leaving
        # "In , is it customary..." — ungrammatical. Attribution is gradient-based
        # and gradients grow with uncertainty, so a malformed prompt inflated the
        # control; both sftdpo conditions had the control out-scoring the real
        # prompt at every layer (negative deltas at 32/32), which normadcontrol
        # never produced.
        #
        # Every control prompt is now identical, so the per-item variation comes
        # only from which label's gradient is taken (yes vs no).
        # lstrip("\n"): the prefix ends in a blank line and the suffix opens with a
        # newline, which the question normally sits between. With no question those
        # collide into an extra empty line that no other prompt in the set has.
        prompt = _CB_FEWSHOT_PREFIX + _CB_PROMPT_SUFFIX.lstrip("\n")
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

    _local = {"normadcontrol", "normad_yn", "culturalbench", "culturalbenchcontrol",
               "blend", "blendcontrol"}
    has_ctrl = "normadcontrol" in dataset_names
    has_yn = "normad_yn" in dataset_names
    has_cb = "culturalbench" in dataset_names
    has_cb_ctrl = "culturalbenchcontrol" in dataset_names
    has_blend = "blend" in dataset_names
    has_blend_ctrl = "blendcontrol" in dataset_names
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
    if has_blend:
        extra_processed += _blend_block(tokenizer, target_data)
    if has_blend_ctrl:
        extra_processed += _blendcontrol_block(tokenizer, target_data)

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
