"""Shared helpers for behavioral evaluation across all four conditions.

Centralizes (a) the condition -> (base_model, adapter_path) resolution and
(b) model loading with QLoRA 4-bit + optional LoRA adapter merge for inference.
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
INSTRUCT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
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
    name: str        # base | sft | dpo | instruct
    base: str        # HF model id for base weights
    adapter: Optional[Path]   # LoRA adapter dir, or None
    final_epoch: Optional[int]  # which epoch's checkpoint to use, or None for default


def resolve_condition(name: str, sft_epoch: int = 3, dpo_epoch: int = 2) -> Condition:
    if name == "base":
        return Condition("base", BASE_MODEL, None, None)
    if name == "instruct":
        return Condition("instruct", INSTRUCT_MODEL, None, None)
    if name == "sft":
        adapter = PROJECT_ROOT / "checkpoints" / "sft" / f"checkpoint-epoch-{sft_epoch}"
        if not adapter.exists():
            adapter = PROJECT_ROOT / "checkpoints" / "sft"
        return Condition("sft", BASE_MODEL, adapter, sft_epoch)
    if name == "dpo":
        adapter = PROJECT_ROOT / "checkpoints" / "dpo" / f"checkpoint-epoch-{dpo_epoch}"
        if not adapter.exists():
            adapter = PROJECT_ROOT / "checkpoints" / "dpo"
        return Condition("dpo", BASE_MODEL, adapter, dpo_epoch)
    raise ValueError(f"unknown condition: {name}")


def load_model_for_eval(cond: Condition):
    """4-bit quantized base + optional LoRA adapter, set to eval mode."""
    tokenizer = AutoTokenizer.from_pretrained(cond.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cond.base,
        quantization_config=bnb,
        device_map="auto",
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
    raw = os.environ.get("CONDITIONS", "base sft dpo instruct")
    return raw.split()
