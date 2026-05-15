"""Sequential SFT+DPO training (C4) — DPO on top of an existing SFT adapter.

Loads Llama 3.2 3B base, attaches the SFT LoRA adapter from C2, merges it into
the base weights (so the DPO step starts from an SFT-fine-tuned model), then
trains a fresh LoRA adapter with the DPO objective on the same HH-RLHF data.

NB on quantization: standalone SFT and DPO use QLoRA 4-bit. This script does
*not*, because `PeftModel.merge_and_unload()` doesn't compose with bitsandbytes
4-bit cleanly (the merge would leave the model in a mixed-precision state). We
load the base in bf16 instead. Llama 3.2 3B in bf16 + DPO training fits within
A5000 24 GB by a comfortable margin, so the practical cost is small.

Reuses `RewardMarginEarlyStop` from `dpo_train` and `load_hh_split` from
`finetune._common` so the C4 condition sees the same 30k examples as C3.

Usage:
    python finetune/sftdpo_train.py --config finetune/configs/sftdpo_config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from finetune._common import load_hh_split
from finetune.dpo_train import RewardMarginEarlyStop


def load_and_merge_sft(cfg: dict):
    """Load Llama base in bf16, attach the C2 SFT adapter, merge it into base weights.

    The returned object is a plain `transformers` model with the SFT contribution
    baked in. DPOTrainer will apply a fresh LoRA on top via `peft_config`.
    """
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    adapter_path = Path(cfg["init_adapter_path"])
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"SFT adapter not found at {adapter_path}. The SFT job must finish "
            "before this script runs — use sbatch --dependency=afterok:<sft_jobid>."
        )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.config.use_cache = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_and_merge_sft(cfg)
    lora_cfg = LoraConfig(**cfg["lora"])
    train_ds = load_hh_split(cfg)

    dpo_args = DPOConfig(
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
        max_length=cfg["max_length"],
        beta=cfg["beta"],
        loss_type=cfg["loss_type"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        bf16=True,
        gradient_checkpointing=True,
    )

    callbacks = []
    if cfg.get("early_stop", {}).get("enabled"):
        callbacks.append(RewardMarginEarlyStop(cfg["early_stop"]["patience_windows"]))

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
