"""Shared utilities for SFT and DPO training scripts.

Lives here so that the SFT chosen-only path and the DPO triple path read HH-RLHF
through the same `split_hh_pair` parser — guaranteeing both conditions train on
identical (prompt, chosen[, rejected]) data and any difference is purely the
training objective.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import BitsAndBytesConfig


HH_PROMPT_RE = re.compile(r"(\n\nHuman: |\n\nAssistant: )")


def split_hh_pair(example: dict) -> dict:
    """HH-RLHF stores chosen/rejected as full transcripts; split into (prompt, chosen, rejected).

    Heuristic: find the last 'Assistant:' marker; everything before is prompt, after is
    response. Both chosen and rejected share the same prompt, so we derive it from one side.
    """
    chosen = example["chosen"]
    rejected = example["rejected"]
    idx_c = chosen.rfind("\n\nAssistant:")
    idx_r = rejected.rfind("\n\nAssistant:")
    if idx_c == -1 or idx_r == -1:
        return {"prompt": "", "chosen": "", "rejected": ""}
    prompt = chosen[: idx_c + len("\n\nAssistant:")]
    return {
        "prompt": prompt,
        "chosen": chosen[idx_c + len("\n\nAssistant:"):].strip(),
        "rejected": rejected[idx_r + len("\n\nAssistant:"):].strip(),
    }


def load_hh_split(cfg: dict):
    """Return a HF Dataset of (prompt, chosen, rejected) triples, filtered + optionally capped.

    Both SFT and DPO use this to ensure they see exactly the same examples in the
    same order (controlled by `seed`). SFT discards the `rejected` field downstream.
    """
    local = Path(cfg["dataset_path"])
    if local.exists() and any(local.iterdir()):
        ds = load_from_disk(str(local))
    else:
        ds = load_dataset(cfg["dataset_name"])
    train = ds["train"]
    mapped = train.map(split_hh_pair, remove_columns=train.column_names)
    filtered = mapped.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    cap = cfg.get("max_train_examples")
    if cap and len(filtered) > cap:
        filtered = filtered.shuffle(seed=cfg["seed"]).select(range(cap))
    return filtered


def build_bnb_config(qcfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
    )
