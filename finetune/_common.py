"""Shared utilities for the DPO training script.

`load_hh_split` and `split_hh_pair` parse HH-RLHF chosen/rejected transcripts into
(prompt, chosen, rejected) triples. `load_coig_p` loads COIG-P preference pairs.
`load_dpo_dataset` dispatches to the right loader based on `dataset_name` in the
config. Used by `finetune/dpo_train.py` and `finetune/sftdpo_train.py`.
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
        "chosen": chosen[idx_c + len("\n\nAssistant:"):],
        "rejected": rejected[idx_r + len("\n\nAssistant:"):],
    }


def load_hh_split(cfg: dict):
    """Return a HF Dataset of (prompt, chosen, rejected) triples, filtered + optionally capped.

    Used by DPO and SFT+DPO to consume HH-RLHF preference data.
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


def _format_conversations(turns: list) -> str:
    """Convert COIG-P conversations list to HH-RLHF-style prompt string.

    Each turn is a dict with 'role' (human/user/assistant) and 'content'.
    Ends with '\n\nAssistant:' so DPOTrainer sees the same split boundary as HH-RLHF.
    """
    parts = []
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role in ("human", "user"):
            parts.append(f"\n\nHuman: {content}")
        elif role == "assistant":
            parts.append(f"\n\nAssistant: {content}")
    parts.append("\n\nAssistant:")
    return "".join(parts)


def load_coig_p(cfg: dict):
    """Load m-a-p/COIG-P preference dataset as (prompt, chosen, rejected) triples.

    COIG-P schema: conversations (list of {role, content}), chosen (dict), rejected (dict).
    We format conversations into a HH-RLHF-style prompt string and extract content from
    chosen/rejected dicts.
    """
    local = Path(cfg["dataset_path"])
    if local.exists() and any(local.iterdir()):
        ds = load_from_disk(str(local))
    else:
        ds = load_dataset(cfg["dataset_name"])
    train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    train = train.filter(lambda x: x.get("domain") == "chat")

    def _normalize(example):
        prompt = _format_conversations(example.get("conversations") or [])
        chosen = example.get("chosen") or {}
        rejected = example.get("rejected") or {}
        return {
            "prompt": prompt,
            "chosen": chosen.get("content", "") if isinstance(chosen, dict) else str(chosen),
            "rejected": rejected.get("content", "") if isinstance(rejected, dict) else str(rejected),
        }

    mapped = train.map(_normalize, remove_columns=train.column_names)
    filtered = mapped.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    cap = cfg.get("max_train_examples")
    if cap and len(filtered) > cap:
        filtered = filtered.shuffle(seed=cfg["seed"]).select(range(cap))
    return filtered


def load_pku_safe_rlhf(cfg: dict):
    """Load PKU-SafeRLHF as (prompt, chosen, rejected) triples.

    Schema: prompt, response_0, response_1, better_response_id (0 or 1).
    We use better_response_id to assign chosen/rejected.
    """
    local = Path(cfg["dataset_path"])
    if local.exists() and any(local.iterdir()):
        ds = load_from_disk(str(local))
    else:
        ds = load_dataset(cfg["dataset_name"])
    train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds

    def _normalize(example):
        bid = example.get("better_response_id")
        if bid == 0:
            chosen, rejected = example["response_0"], example["response_1"]
        elif bid == 1:
            chosen, rejected = example["response_1"], example["response_0"]
        else:
            chosen, rejected = "", ""
        return {"prompt": example.get("prompt", ""), "chosen": chosen, "rejected": rejected}

    mapped = train.map(_normalize, remove_columns=train.column_names)
    filtered = mapped.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])
    cap = cfg.get("max_train_examples")
    if cap and len(filtered) > cap:
        filtered = filtered.shuffle(seed=cfg["seed"]).select(range(cap))
    return filtered


def load_dpo_dataset(cfg: dict):
    """Dispatch to the right preference-data loader based on dataset_name."""
    name = cfg.get("dataset_name", "")
    if "hh-rlhf" in name or "hh_rlhf" in name:
        return load_hh_split(cfg)
    if "COIG-P" in name or "coig-p" in name.lower():
        return load_coig_p(cfg)
    if "PKU-SafeRLHF" in name or "pku-safe" in name.lower() or "pku_safe" in name.lower():
        return load_pku_safe_rlhf(cfg)
    raise ValueError(
        f"Unsupported DPO dataset: {name!r}. "
        f"Supported: Anthropic/hh-rlhf, m-a-p/COIG-P, PKU-Alignment/PKU-SafeRLHF-single-dimension"
    )


def build_bnb_config(qcfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
    )
