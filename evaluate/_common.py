"""Shared helpers for behavioral evaluation across all conditions.

Centralizes (a) the condition -> (base_model, adapter_path) resolution and
(b) model loading for inference (matched-bf16 by default, qlora_4bit available
as a backward-compatible fallback).

Conditions: base (C1), sft (C2), dpo (C3), sftdpo (C4).
SFT uses Alpaca; DPO uses HH-RLHF preferences; C4 = DPO on top of merged C2.

Model sizes: '3b' (Llama 3.2 3B, default), '8b' (Llama 3.1 8B), or 'gemma4'
(Gemma 4 12B). When model_size != '3b', checkpoint dirs and output paths are
suffixed with the size label (e.g. checkpoints/sft_8b, outputs/neurons/sft_8b/)
so that 3B artifacts on disk are never overwritten by larger-model runs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Mapping from size label to HF model id. Extend this dict to add new models.
MODEL_REGISTRY = {
    "3b": "meta-llama/Llama-3.2-3B",
    "8b": "meta-llama/Llama-3.1-8B",
    "gemma4": "google/gemma-4-12B",
    "qwen35": "Qwen/Qwen3.5-9B-Base",
}
DEFAULT_MODEL_SIZE = "3b"

# Backward-compat alias; existing imports of BASE_MODEL keep working as 3B.
BASE_MODEL = MODEL_REGISTRY[DEFAULT_MODEL_SIZE]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def model_id_for(size: str) -> str:
    """Return the HF model id for a size label, or raise if unknown."""
    if size not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model_size {size!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[size]


def size_suffix(size: str) -> str:
    """Return '' for 3B (back-compat) or '_{size}' for everything else.

    Used to keep existing 3B paths (checkpoints/sft, outputs/neurons/sft/)
    unchanged while suffixing larger-model artifacts so they coexist on disk.
    """
    return "" if size == DEFAULT_MODEL_SIZE else f"_{size}"


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
    name: str        # base | sft | dpo | sftdpo
    base: str        # HF model id for base weights
    adapter: Optional[Path]                          # primary LoRA adapter, applied last
    pre_merge_adapter: Optional[Path] = None         # adapter merged into base BEFORE primary
    final_epoch: Optional[int] = None
    model_size: str = DEFAULT_MODEL_SIZE              # '3b' | '8b' — used for output-path suffixing


def _latest_checkpoint(parent: Path) -> Path:
    """Pick the highest-numbered `checkpoint-N` subdir, falling back to `parent` itself."""
    candidates = sorted(parent.glob("checkpoint-*"),
                        key=lambda p: int(p.name.rsplit("-", 1)[-1]) if p.name.rsplit("-", 1)[-1].isdigit() else -1)
    return candidates[-1] if candidates else parent


def resolve_condition(name: str, sft_epoch: int = 3, dpo_epoch: int = 2,
                      model_size: str = DEFAULT_MODEL_SIZE) -> Condition:
    base = model_id_for(model_size)
    sfx = size_suffix(model_size)
    ckpt_root = PROJECT_ROOT / "checkpoints"

    if name == "base":
        return Condition("base", base, None, model_size=model_size)
    if name == "dpo":
        adapter = _latest_checkpoint(ckpt_root / f"dpo{sfx}")
        return Condition("dpo", base, adapter, final_epoch=dpo_epoch, model_size=model_size)
    if name == "dpo_coig":
        adapter = _latest_checkpoint(ckpt_root / f"dpo_coig{sfx}")
        return Condition("dpo_coig", base, adapter, final_epoch=dpo_epoch, model_size=model_size)
    if name == "dpo_pku":
        adapter = _latest_checkpoint(ckpt_root / f"dpo_pku{sfx}")
        return Condition("dpo_pku", base, adapter, final_epoch=dpo_epoch, model_size=model_size)
    if name == "sft":
        adapter = _latest_checkpoint(ckpt_root / f"sft{sfx}")
        return Condition("sft", base, adapter, final_epoch=sft_epoch, model_size=model_size)
    if name == "sftdpo":
        # C4 — DPO trained on top of the merged SFT adapter.
        sft_adapter = _latest_checkpoint(ckpt_root / f"sft{sfx}")
        sftdpo_adapter = _latest_checkpoint(ckpt_root / f"sftdpo{sfx}")
        return Condition(
            "sftdpo", base,
            adapter=sftdpo_adapter,
            pre_merge_adapter=sft_adapter,
            final_epoch=dpo_epoch,
            model_size=model_size,
        )
    if name == "sftdpo_coig":
        sft_adapter = _latest_checkpoint(ckpt_root / f"sft{sfx}")
        sftdpo_adapter = _latest_checkpoint(ckpt_root / f"sftdpo_coig{sfx}")
        return Condition(
            "sftdpo_coig", base,
            adapter=sftdpo_adapter,
            pre_merge_adapter=sft_adapter,
            final_epoch=dpo_epoch,
            model_size=model_size,
        )
    if name == "sftdpo_pku":
        sft_adapter = _latest_checkpoint(ckpt_root / f"sft{sfx}")
        sftdpo_adapter = _latest_checkpoint(ckpt_root / f"sftdpo_pku{sfx}")
        return Condition(
            "sftdpo_pku", base,
            adapter=sftdpo_adapter,
            pre_merge_adapter=sft_adapter,
            final_epoch=dpo_epoch,
            model_size=model_size,
        )
    raise ValueError(f"unknown condition: {name}")


def load_model_for_eval(cond: Condition, precision: str = "matched_bf16"):
    """Load base + optional pre-merge adapter + optional primary adapter, set to eval mode.

    `precision` controls quantization regime across conditions:

      - 'matched_bf16' (default, recommended): every condition loads the base
        in bf16. This eliminates the cross-condition precision confound — C4
        always had to be bf16 (since merge_and_unload() doesn't compose with
        4-bit), and historically C1/C2/C3 used 4-bit, so attribution scores
        across conditions were computed in different numerical regimes. With
        'matched_bf16', all four conditions use the same precision.
        Llama 3.2 3B in bf16 ≈ 6 GB, Llama 3.1 8B in bf16 ≈ 16 GB; both fit
        comfortably on a 24 GB A5000.

      - 'qlora_4bit' (legacy): conditions without a pre-merge adapter
        (C1/C2/C3) load in 4-bit NF4; C4 still uses bf16 because of the
        merge constraint. This is the original behavior used by the May-2026
        result set. Kept for backward compatibility / robustness checks
        ("does the headline finding survive both precision regimes?").
    """
    tokenizer = AutoTokenizer.from_pretrained(cond.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decide whether to use 4-bit. The merge path always forces bf16; otherwise
    # we honor `precision`.
    use_4bit = (
        precision == "qlora_4bit"
        and cond.pre_merge_adapter is None
    )

    if use_4bit:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cond.base, torch_dtype=torch.bfloat16, device_map="auto",
        )

    if cond.pre_merge_adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.pre_merge_adapter))
        model = model.merge_and_unload()

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
    raw = os.environ.get("CONDITIONS", "base sft dpo sftdpo")
    return raw.split()
