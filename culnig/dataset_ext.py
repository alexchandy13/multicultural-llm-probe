"""Extends upstream's `load_dataset_neuron_scores` with a `normadcontrol` block.

Upstream's `culnig/_upstream/dataset.py` already supports `normad` (lines 75-156)
but only stubs out `normadcontrol` in its `__main__` test (line 862). The plan's
Step 2 fills that gap: NormAd, but with the country field removed from the prompt
and culture markers regex-stripped from the story.

We do this by monkey-patching upstream's loader before any caller imports it.
Import this module first (or set `PYTHONPATH` to include `culnig/`); the patch is
idempotent.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

UPSTREAM = Path(__file__).resolve().parent / "_upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

import dataset as _upstream_dataset  # noqa: E402
from datasets import load_dataset  # noqa: E402

from culnig.construct_normad_ctrl import strip_culture  # noqa: E402


NORMAD_CTRL_PROMPT = (
    "You will be given a short story involving 2 or more characters about everyday "
    "situations. Read the story and determine if the character's behavior aligns with "
    "common social norms.\nRespond only with the number 1, 2, or 3.\n\n"
    "Story: {story}\nOptions: {options}\n"
)


_original_loader = _upstream_dataset.load_dataset_neuron_scores
_patched = False


def _normadcontrol_block(tokenizer, target_countries, target_data):
    """Build the processed dataset for `normadcontrol`.

    Mirrors upstream's `normad` block but:
      - Drops the country field from the prompt template.
      - Strips culture markers from the story via `strip_culture`.
      - Tags `dataset_name='normadcontrol'` so the decide-script suffix `control` matches.
    """
    c2n = _upstream_dataset.COUNTRY_TO_NAME["normad"]
    rev_c2n = {v: k for k, v in c2n.items()}

    dataset = load_dataset("akhilayerukola/NormAd", split="train")

    if target_countries is not None:
        target = [c2n[c] for c in target_countries]
        dataset = dataset.filter(lambda x: x["Country"] in target)
    else:
        target = np.unique(dataset["Country"]).tolist()

    if target_data != "all":
        ids = []
        for country in target:
            cdata = dataset.filter(lambda x, c=country: x["Country"] == c)
            for label in ["yes", "no", "neutral"]:
                ldata = cdata.filter(lambda x, l=label: x["Gold Label"] == l)
                n = len(ldata)
                half = n // 2
                if target_data == "neuron":
                    ids.extend(ldata.select(range(half))["ID"])
                else:
                    ids.extend(ldata.select(range(half, n))["ID"])
        dataset = dataset.filter(lambda x: x["ID"] in ids)

    instructions = [NORMAD_CTRL_PROMPT]

    def make_preprocess(instruction, inst_idx):
        def preprocess(examples):
            hash_input = f"{examples['ID']}_{instruction}"
            seed = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16) % (2 ** 32)
            rng = random.Random(seed)

            options = [1, 2, 3]
            rng.shuffle(options)
            option_labels = sorted(
                {options[0]: "yes", options[1]: "no", options[2]: "neutral"}.items()
            )
            option_str = ", ".join(f"{k}: {v}" for k, v in option_labels)

            gold = examples["Gold Label"]
            if gold == "yes":
                label = options[0]
            elif gold == "no":
                label = options[1]
            elif gold == "neutral":
                label = options[2]
            else:
                raise ValueError(f"Unknown gold label: {gold}")

            stripped_story = strip_culture(examples["Story"])
            input_text = instruction.format(story=stripped_story, options=option_str)
            # No chat template: caller strips tokenizer.chat_template to None so
            # all four conditions (C1 base, C2 SFT, C3 DPO, C4 Instruct) see the
            # same raw-text prompt format. Without this, C4 would be chat-formatted
            # and C1-C3 raw-text, invalidating cross-condition comparisons.
            tokenized = tokenizer(
                input_text, return_tensors="pt", add_special_tokens=True
            )
            return {
                "input_text": input_text,
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "label": str(label),
                "country": rev_c2n.get(examples["Country"], examples["Country"]),
                "id": str(examples["ID"]),
                "instruction_id": inst_idx,
                "dataset_name": "normadcontrol",
                "options": [str(o) for o in options],
            }
        return preprocess

    processed = []
    for inst_idx, instruction in enumerate(instructions):
        processed.append(
            dataset.map(
                make_preprocess(instruction, inst_idx),
                remove_columns=dataset.column_names,
                num_proc=1,
            )
        )
    return processed


def _patched_loader(dataset_names, tokenizer, batch_size,
                    target_countries=None, target_data="all"):
    """Drop-in replacement: handle `normadcontrol`, then delegate everything else."""
    from datasets import concatenate_datasets
    import torch
    from torch.nn.utils.rnn import pad_sequence

    has_ctrl = "normadcontrol" in dataset_names
    remaining = [d for d in dataset_names if d != "normadcontrol"]
    extra_processed = (
        _normadcontrol_block(tokenizer, target_countries, target_data) if has_ctrl else []
    )

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
