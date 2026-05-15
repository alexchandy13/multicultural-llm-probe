"""SFT on HH-RLHF chosen responses with QLoRA 4-bit + LoRA adapters.

NOTE: SFT now uses HH-RLHF, not Alpaca. The plan originally specified Alpaca for SFT
and HH-RLHF for DPO, but that conflated two confounds: training method AND training
data. To isolate the SFT-vs-DPO contrast at the heart of the RQ, both conditions
now read the same HH-RLHF data via `load_hh_split` — SFT keeps only the (prompt,
chosen) pairs and trains the standard next-token-prediction objective, while DPO
uses the full (prompt, chosen, rejected) triples with the DPO objective. The same
shuffle seed + max_train_examples cap is used in both configs so they literally
see the same 30k examples in the same order.

Usage:
    python finetune/sft_train.py --config finetune/configs/sft_config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from finetune._common import build_bnb_config, load_hh_split


def load_sft_chosen(cfg: dict):
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

    train_ds = load_sft_chosen(cfg)

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
