"""Shared helpers for behavioral evaluation across all conditions.

Centralizes (a) the condition -> (base_model, adapter_path) resolution and
(b) model loading with QLoRA 4-bit + optional LoRA adapter merge for inference.

Conditions: base (C1), sft_alpaca (C2), dpo (C3), sftdpo_alpaca (C4).
SFT uses Alpaca; DPO uses HH-RLHF preferences; C4 = DPO on top of merged C2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Expanded from the plan's 12-country list to ~50 of NormAd's 75 countries.
# The plan's 12 are kept as a strict subset (see paper text). Cultural groupings
# follow conventional anglosphere + Western/Northern Europe definitions for
# Western, and Asia/MENA/Sub-Saharan Africa/Latin America for Non-Western.
# Contested cases left in "Other": Eastern Europe, Russia/Ukraine, Israel/Türkiye,
# Pacific Islands, Mauritius — to keep the W vs NW contrast clean.
WESTERN = {
    # Plan's original 5
    "US", "UK", "Germany", "Spain", "Australia",
    # Anglosphere
    "Canada", "Ireland", "New_Zealand",
    # Western / Northern Europe
    "France", "Italy", "Netherlands", "Austria", "Sweden", "Portugal", "Greece",
}
NON_WESTERN = {
    # Plan's original 8 (Nigeria absent from NormAd)
    "Japan", "China", "India", "Iran", "Indonesia",
    "Nigeria", "Mexico", "South_Korea", "South Korea",
    # South Asia
    "Pakistan", "Bangladesh", "Sri_Lanka", "Nepal", "Afghanistan",
    # Southeast Asia
    "Thailand", "Vietnam", "Philippines", "Malaysia", "Singapore",
    "Cambodia", "Laos", "Myanmar",
    # East Asia
    "Hong_Kong", "Taiwan",
    # Middle East / North Africa
    "Egypt", "Lebanon", "Iraq", "Syria", "Saudi_Arabia",
    # Sub-Saharan Africa
    "Ethiopia", "Kenya", "South_Africa",
    # Latin America
    "Brazil", "Argentina", "Chile", "Colombia", "Peru",
}


@dataclass
class Condition:
    name: str        # base | sft_alpaca | dpo | sftdpo_alpaca
    base: str        # HF model id for base weights
    adapter: Optional[Path]                          # primary LoRA adapter, applied last
    pre_merge_adapter: Optional[Path] = None         # adapter merged into base BEFORE primary
    final_epoch: Optional[int] = None


def _latest_checkpoint(parent: Path) -> Path:
    """Pick the highest-numbered `checkpoint-N` subdir, falling back to `parent` itself."""
    candidates = sorted(parent.glob("checkpoint-*"),
                        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1)
    return candidates[-1] if candidates else parent


def resolve_condition(name: str, sft_epoch: int = 3, dpo_epoch: int = 2) -> Condition:
    if name == "base":
        return Condition("base", BASE_MODEL, None)
    if name == "dpo":
        adapter = _latest_checkpoint(PROJECT_ROOT / "checkpoints" / "dpo")
        return Condition("dpo", BASE_MODEL, adapter, final_epoch=dpo_epoch)
    if name == "sft_alpaca":
        # C2 — Alpaca-trained SFT adapter (only SFT variant retained).
        adapter = _latest_checkpoint(PROJECT_ROOT / "checkpoints" / "sft_alpaca")
        return Condition("sft_alpaca", BASE_MODEL, adapter, final_epoch=sft_epoch)
    if name == "sftdpo_alpaca":
        # C4 — DPO trained on top of the merged Alpaca SFT adapter. For correct
        # inference we merge the SFT adapter into the base first, then apply
        # the sftdpo adapter on top (its weights are deltas from base + SFT).
        sft_adapter = _latest_checkpoint(PROJECT_ROOT / "checkpoints" / "sft_alpaca")
        sftdpo_adapter = _latest_checkpoint(PROJECT_ROOT / "checkpoints" / "sftdpo_alpaca")
        return Condition(
            "sftdpo_alpaca", BASE_MODEL,
            adapter=sftdpo_adapter,
            pre_merge_adapter=sft_adapter,
            final_epoch=dpo_epoch,
        )
    raise ValueError(f"unknown condition: {name}")


def load_model_for_eval(cond: Condition):
    """Load base + optional pre-merge adapter + optional primary adapter, set to eval mode.

    Quantization rule:
      - No pre_merge_adapter → load base in 4-bit (QLoRA-style, low memory).
      - pre_merge_adapter set → load base in bf16 because `merge_and_unload()`
        doesn't compose with bitsandbytes 4-bit. Llama 3.2 3B in bf16 (~6 GB)
        plus a LoRA adapter still fits comfortably on A5000 24 GB for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(cond.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cond.pre_merge_adapter is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cond.base, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(model, str(cond.pre_merge_adapter))
        model = model.merge_and_unload()
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cond.base, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    if cond.adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.adapter))
    model.eval()
    return tokenizer, model


def culture_group(country: str) -> str:
    if country in WESTERN:
        return "Western"
    if country in NON_WESTERN:
        return "Non-Western"
    return "Other"


def conditions_from_env() -> list[str]:
    raw = os.environ.get("CONDITIONS", "base sft_alpaca dpo sftdpo_alpaca")
    return raw.split()
