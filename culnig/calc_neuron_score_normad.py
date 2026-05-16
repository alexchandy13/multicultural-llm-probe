"""Fork of upstream `CULNIG/calc_neuron_score.py` with two minimal additions:

  1. Llama-3.2-3B (base + Instruct) added to the model whitelist.
  2. QLoRA 4-bit base loading + optional LoRA adapter for our SFT/DPO conditions.

The core gradient-scoring loop (`calculate_scores`) is imported from upstream
unchanged, per the plan's directive: "keep gradient scoring logic completely
unchanged — no modifications to the core algorithm."

Outputs land in our project's outputs/neurons/ tree instead of upstream's
default ../outputs/, so analysis scripts can find them.

Usage:
    python culnig/calc_neuron_score_normad.py \
        --condition sft --dataset-names normad
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = PROJECT_ROOT / "culnig" / "_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PROJECT_ROOT))

# Install our normadcontrol patch on upstream's dataset module BEFORE anything
# else imports it.
import culnig.dataset_ext  # noqa: F401, E402

# Now safe to import upstream pieces.
from CULNIG import calc_neuron_score as upstream_score  # noqa: E402

from evaluate._common import resolve_condition  # noqa: E402


LLAMA_32_MODELS = {"meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"}
LLAMA_31_BRANCH = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = upstream_score.BATCH_SIZE


def _pin_name_or_path(model):
    """Permanently set model.name_or_path so upstream's hard-coded whitelists accept it.

    Upstream reads model.name_or_path in three hot paths:
      - utils.get_target_module (line 16): exact-match whitelist.
      - utils.get_text_model (line 153): substring check — matches "Llama-3" already,
        so we don't strictly need to swap for this one, but a single value keeps
        downstream behavior uniform.
      - CULNIG/calc_neuron_score.calculate_scores (lines 118/121): exact-match whitelist
        INSIDE the per-batch / per-module loops. Hit on every forward.

    Llama-3.2-3B is not in those exact-match lists. Its MLP/attention module names
    are identical to Llama-3.1-8B-Instruct (same `q_proj`/`k_proj`/`v_proj`/`o_proj`/
    `gate_proj`/`up_proj`/`down_proj` under `model.layers[i].mlp` and `.self_attn`),
    so we pin name_or_path to the 3.1 string and reuse that branch.

    Note: this is a **permanent** swap, not scoped. The reads happen inside loops,
    so any swap-and-restore wrapper would have to bracket every read site —
    pinning once at load is strictly simpler and equally correct. The original
    value isn't preserved because nothing downstream needs it.
    """
    if model.name_or_path in LLAMA_32_MODELS:
        model.name_or_path = LLAMA_31_BRANCH


def load_model_for_culnig(condition_name: str):
    """Load base + optional pre-merge adapter + optional primary adapter, merging both.

    Quantization rule:
      - No pre_merge_adapter → 4-bit base (QLoRA-style, low memory).
      - pre_merge_adapter set (C4 sftdpo) → bf16 base, because `merge_and_unload()`
        doesn't compose with bitsandbytes 4-bit. Llama 3.2 3B in bf16 fits in
        A5000 24 GB for gradient scoring (forward + backward on adapter-merged
        base; no optimizer state, no LoRA params).
    """
    cond = resolve_condition(condition_name)
    tokenizer = AutoTokenizer.from_pretrained(cond.base, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Strip the chat template so every condition produces identical raw-text prompts.
    # C5 (Instruct) ships a chat template; C1-C4 (base + adapters) don't. If we let
    # the template apply on C5, its prompts become chat-formatted and gradient scores
    # are no longer comparable across conditions. Forcing chat_template=None makes
    # upstream's `try: apply_chat_template ... except: pass` blocks fall back to
    # raw text uniformly for every dataset (normad, normadcontrol, blend, etc.).
    tokenizer.chat_template = None

    if cond.pre_merge_adapter is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cond.base,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, str(cond.pre_merge_adapter))
        model = model.merge_and_unload()
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cond.base,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    if cond.adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.adapter))
        model = model.merge_and_unload()  # merge so gradients flow into base weights

    # See _pin_name_or_path docstring for why this is a permanent (not scoped) swap.
    _pin_name_or_path(model)
    return model, tokenizer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def run(condition_name: str, dataset_names: list[str], out_root: Path, logger):
    model, tokenizer = load_model_for_culnig(condition_name)
    logger.info(f"Loaded model for condition={condition_name} on {model.device}")

    dataset_names = sorted(dataset_names)

    # Main dataset(s)
    dataloader = upstream_score.load_dataset_neuron_scores(
        dataset_names, tokenizer, batch_size=BATCH_SIZE,
        target_countries=None, target_data="neuron",
    )
    raw_scores, total_probs = upstream_score.calculate_scores(
        model, tokenizer, dataloader, logger
    )
    neuron_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (module_name, layer_idx, neuron_idx), per_country in raw_scores.items():
        key = f"{module_name}_{layer_idx}_{neuron_idx}"
        for country, score in per_country.items():
            neuron_scores[key][country] += score

    dataset_ids = defaultdict(list)
    for item in dataloader.dataset:
        if item["id"] not in dataset_ids[item["dataset_name"]]:
            dataset_ids[item["dataset_name"]].append(item["id"])

    out_dir = out_root / condition_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{''.join(dataset_names)}_max_scores.json"
    out_file.write_text(json.dumps({
        "neuron_scores": neuron_scores,
        "total_probabilities_per_country": dict(total_probs),
        "dataset_ids": dict(dataset_ids),
    }, indent=2))
    logger.info(f"Wrote {out_file}")

    # CountryRC second pass — same scoring loop, target countries restricted.
    crc_dataloader = upstream_score.load_dataset_neuron_scores(
        dataset_names=["countryrc"], tokenizer=tokenizer, batch_size=BATCH_SIZE,
        target_countries=upstream_score.TARGET_COUNTRIES, target_data="neuron",
    )
    crc_raw, crc_probs = upstream_score.calculate_scores(
        model, tokenizer, crc_dataloader, logger
    )
    crc_scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for (module_name, layer_idx, neuron_idx), per_country in crc_raw.items():
        key = f"{module_name}_{layer_idx}_{neuron_idx}"
        for country, score in per_country.items():
            crc_scores[key][country] += score
    crc_ids = defaultdict(list)
    for item in crc_dataloader.dataset:
        if item["id"] not in crc_ids[item["dataset_name"]]:
            crc_ids[item["dataset_name"]].append(item["id"])

    crc_file = out_dir / "countryrc_max_scores.json"
    crc_file.write_text(json.dumps({
        "neuron_scores": crc_scores,
        "total_probabilities_per_country": dict(crc_probs),
        "dataset_ids": dict(crc_ids),
    }, indent=2))
    logger.info(f"Wrote {crc_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["base", "sft", "dpo", "sftdpo", "instruct",
                                 "sft_alpaca", "sftdpo_alpaca"])
    parser.add_argument("--dataset-names", nargs="+", required=True,
                        help="e.g. `normad` or `normadcontrol` (single name per run).")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs" / "neurons"))
    return parser.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    logger = setup_logging()
    run(args.condition, args.dataset_names, Path(args.out_root), logger)


if __name__ == "__main__":
    main()
