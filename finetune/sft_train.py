"""SFT with QLoRA 4-bit + LoRA adapters, dataset-driven.

Three supported source datasets:
  - `Anthropic/hh-rlhf` (primary):  SFT on the chosen response of each pair, so SFT
    and DPO read identical source data and any C2-vs-C3 contrast isolates the
    training objective. See `load_hh_split` in finetune/_common.py.
  - `tatsu-lab/alpaca` (robustness variant): SFT on generic instruction-response
    pairs. Used for the C2a/C4a Alpaca robustness check — same training pipeline,
    different SFT source.
  - `GAIR/lima` (second robustness variant): 1000 high-quality human-curated
    instruction-response pairs from Zhou et al. 2023. Used for the C2b/C4b LIMA
    robustness check — tests whether the SFT effect on cultural neurons holds
    under a small, quality-curated alternative to Alpaca's GPT-3.5-generated data.

The dataset is selected by the `dataset_name` field in the config — no CLI flag or
script branching at the call site. Existing HH-RLHF / Alpaca behavior is unchanged.

Usage:
    python finetune/sft_train.py --config finetune/configs/sft_config.yaml         # HH-RLHF
    python finetune/sft_train.py --config finetune/configs/sft_alpaca_config.yaml  # Alpaca
    python finetune/sft_train.py --config finetune/configs/sft_lima_config.yaml    # LIMA
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from finetune._common import build_bnb_config, load_hh_split


ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def _format_alpaca(example: dict) -> dict:
    if example.get("input"):
        text = ALPACA_TEMPLATE_WITH_INPUT.format(**example)
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(**example)
    return {"text": text}


def _load_alpaca(cfg: dict):
    local = Path(cfg["dataset_path"])
    if local.exists() and any(local.iterdir()):
        ds = load_from_disk(str(local))
    else:
        ds = load_dataset(cfg["dataset_name"])
    train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    return train.map(_format_alpaca, remove_columns=train.column_names)


def _load_hh_chosen(cfg: dict):
    """Load HH-RLHF and project to a single `text` field = '{prompt} {chosen}'.

    HH-RLHF prompts already terminate with '\\n\\nAssistant:', so concatenating
    with a space yields a well-formed transcript ending in the chosen response —
    exactly what an SFT next-token loss should consume.
    """
    triples = load_hh_split(cfg)
    return triples.map(
        lambda x: {"text": f"{x['prompt']} {x['chosen']}"},
        remove_columns=triples.column_names,
    )


# LIMA stores each example as `conversations`: a list with even length, alternating
# [user_msg, assistant_response, user_msg, assistant_response, ...]. Most rows are
# single-turn (2 elements); a small fraction are multi-turn. We render to plain
# instruction/response text using an Alpaca-style template, then concatenate
# additional turns for multi-turn rows.
LIMA_FIRST_TURN_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
)
LIMA_CONTINUATION_TEMPLATE = (
    "\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def _format_lima(example: dict) -> dict:
    convs = example.get("conversations") or []
    # Even length required (paired user/assistant turns). Skip malformed rows.
    if not convs or len(convs) % 2 != 0:
        return {"text": ""}
    text = LIMA_FIRST_TURN_TEMPLATE.format(instruction=convs[0], output=convs[1])
    for i in range(2, len(convs), 2):
        text += LIMA_CONTINUATION_TEMPLATE.format(instruction=convs[i], output=convs[i + 1])
    return {"text": text}


def _load_lima(cfg: dict):
    local = Path(cfg["dataset_path"])
    if local.exists() and any(local.iterdir()):
        ds = load_from_disk(str(local))
    else:
        ds = load_dataset(cfg["dataset_name"])
    train = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
    formatted = train.map(_format_lima, remove_columns=train.column_names)
    # Drop any rows that hit the malformed-skip path above.
    return formatted.filter(lambda x: bool(x["text"]))


def load_train_dataset(cfg: dict):
    """Dispatch on `dataset_name` to pick the right loader + text formatting."""
    name = cfg.get("dataset_name", "")
    if name.startswith("tatsu-lab/alpaca") or name == "alpaca":
        return _load_alpaca(cfg)
    if name.startswith("Anthropic/hh-rlhf") or name == "hh-rlhf":
        return _load_hh_chosen(cfg)
    if name.startswith("GAIR/lima") or name == "lima":
        return _load_lima(cfg)
    raise ValueError(
        f"Unknown SFT dataset: {name!r}. "
        "Supported: tatsu-lab/alpaca, Anthropic/hh-rlhf, GAIR/lima."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = build_bnb_config(cfg["quantization"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(**cfg["lora"])

    train_ds = load_train_dataset(cfg)

    sft_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        max_length=cfg["max_seq_length"],  # TRL renamed this kwarg
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,  # TRL/transformers renamed `tokenizer`
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
