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


# Models whose architecture matches Llama-3.1-8B-Instruct module-for-module
# (same q/k/v/o_proj + gate/up/down_proj names under the same parent path).
# Upstream CULNIG's hard-coded whitelist only accepts a few exact strings; we
# pin every member of this set to LLAMA_31_BRANCH at load time so the
# upstream branch fires for all of them.
PIN_TO_LLAMA_31_BRANCH = {
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
}
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
    if model.name_or_path in PIN_TO_LLAMA_31_BRANCH:
        model.name_or_path = LLAMA_31_BRANCH


def load_model_for_culnig(condition_name: str, model_size: str = "3b",
                          precision: str = "matched_bf16"):
    """Load base + optional pre-merge adapter + optional primary adapter, merging both.

    `precision` controls quantization regime (same semantics as
    evaluate._common.load_model_for_eval):
      - 'matched_bf16' (default): every condition in bf16. Eliminates the
        precision confound that arises from the C4 merge step forcing bf16
        while C1/C2/C3 could otherwise use 4-bit. Use this for any cross-
        condition mechanistic analysis (Jaccard, attribution comparisons).
      - 'qlora_4bit': C1/C2/C3 in 4-bit, C4 in bf16 (original behavior).

    Memory: at 3B bf16 ≈ 6 GB; at 8B bf16 ≈ 16 GB. Both fit on A5000 24 GB
    for gradient scoring (forward + backward on the adapter-merged base).
    """
    cond = resolve_condition(condition_name, model_size=model_size)
    tokenizer = AutoTokenizer.from_pretrained(cond.base, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Strip the chat template so every condition produces identical raw-text prompts.
    # If a condition ships a chat template (e.g. an Instruct variant), its prompts
    # become chat-formatted and gradient scores are no longer comparable across
    # conditions. Forcing a pass-through template makes upstream's
    # `try: apply_chat_template ... except: pass` blocks fall back to raw text
    # uniformly for every dataset (normad, normadcontrol, blend, etc.).
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    use_4bit = (
        precision == "qlora_4bit"
        and cond.pre_merge_adapter is None
    )

    if use_4bit:
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cond.base,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    if cond.pre_merge_adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.pre_merge_adapter))
        model = model.merge_and_unload()

    if cond.adapter is not None:
        model = PeftModel.from_pretrained(model, str(cond.adapter))
        model = model.merge_and_unload()  # merge so gradients flow into base weights

    # The base model loads with all parameters frozen (4-bit weights have
    # requires_grad=False; PEFT inference-mode merges also leave the result
    # frozen). CULNIG's per-layer hooks call `output.retain_grad()`, which
    # requires the activation to have requires_grad=True — that in turn
    # requires the input embedding's output to carry grad. enable_input_require_grads()
    # adds the forward hook on the embedding layer that makes this work without
    # us needing to flip any parameter's requires_grad. This is the same trick
    # `prepare_model_for_kbit_training` uses internally.
    model.enable_input_require_grads()

    # See _pin_name_or_path docstring for why this is a permanent (not scoped) swap.
    _pin_name_or_path(model)
    return model, tokenizer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def run(condition_name: str, dataset_names: list[str], out_root: Path, logger,
        model_size: str = "3b", precision: str = "matched_bf16"):
    model, tokenizer = load_model_for_culnig(
        condition_name, model_size=model_size, precision=precision
    )
    logger.info(f"Loaded model for condition={condition_name} "
                f"(model_size={model_size}, precision={precision}) on {model.device}")

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

    # Suffix the per-condition output dir with the model size so different
    # base models don't clobber each other on disk (e.g. outputs/neurons/sft_8b/).
    size_sfx = "" if model_size == "3b" else f"_{model_size}"
    out_dir = out_root / f"{condition_name}{size_sfx}"
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
                        choices=["base", "dpo", "sft", "sftdpo"])
    parser.add_argument("--dataset-names", nargs="+", required=True,
                        help="e.g. `normad` or `normadcontrol` (single name per run).")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "outputs" / "neurons"))
    parser.add_argument(
        "--model-size", default="3b", choices=["3b", "8b"],
        help="Base model size. '3b'=Llama-3.2-3B (default), '8b'=Llama-3.1-8B. "
             "Per-condition output dir is suffixed with the size when not 3b.",
    )
    parser.add_argument(
        "--precision", default="matched_bf16",
        choices=["matched_bf16", "qlora_4bit"],
        help="'matched_bf16' (default): all conditions in bf16. 'qlora_4bit': "
             "C1/C2/C3 in 4-bit (legacy regime).",
    )
    return parser.parse_args()


def main():
    set_seed(42)
    args = parse_args()
    logger = setup_logging()
    run(
        args.condition, args.dataset_names, Path(args.out_root), logger,
        model_size=args.model_size, precision=args.precision,
    )


if __name__ == "__main__":
    main()
