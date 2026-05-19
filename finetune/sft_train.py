"""SFT with QLoRA 4-bit + LoRA adapters, on the Alpaca instruction dataset.

The only supported SFT source is `tatsu-lab/alpaca` — generic English instruction-
response pairs used for the SFT condition (C2). The DPO condition (C3) uses
HH-RLHF preference pairs via the separate `finetune/dpo_train.py` script.

Usage:
    python finetune/sft_train.py --config finetune/configs/sft_alpaca_config.yaml
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

from finetune._common import build_bnb_config


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


def load_train_dataset(cfg: dict):
    name = cfg.get("dataset_name", "")
    if name.startswith("tatsu-lab/alpaca") or name == "alpaca":
        return _load_alpaca(cfg)
    raise ValueError(
        f"Unsupported SFT dataset: {name!r}. Only tatsu-lab/alpaca is supported."
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
