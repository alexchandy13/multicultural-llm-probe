"""DPO on Anthropic HH-RLHF with QLoRA 4-bit + LoRA adapters.

Reads chosen/rejected pairs, applies the same QLoRA setup as SFT, and trains the policy
with TRL's DPOTrainer. Includes a reward-margin early-stop hook (plan §"Risk Mitigation":
if epoch 2 degrades, fall back to epoch 1).

Usage:
    python finetune/dpo_train.py --config finetune/configs/dpo_config.yaml
"""
from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer


HH_PROMPT_RE = re.compile(r"(\n\nHuman: |\n\nAssistant: )")


def split_hh_pair(example: dict) -> dict:
    """HH-RLHF stores chosen/rejected as full transcripts; split into (prompt, chosen, rejected).

    Heuristic: find the last 'Assistant:' marker; everything before is prompt, after is response.
    Both chosen and rejected share the same prompt, so we derive it from one side.
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
        "chosen": chosen[idx_c + len("\n\nAssistant:"):].strip(),
        "rejected": rejected[idx_r + len("\n\nAssistant:"):].strip(),
    }


def load_hh(cfg: dict):
    local = Path(cfg["dataset_path"])
    if local.exists() and any(local.iterdir()):
        ds = load_from_disk(str(local))
    else:
        ds = load_dataset(cfg["dataset_name"])
    train = ds["train"]
    mapped = train.map(split_hh_pair, remove_columns=train.column_names)
    return mapped.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"])


def build_bnb_config(qcfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=qcfg["load_in_4bit"],
        bnb_4bit_quant_type=qcfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=qcfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, qcfg["bnb_4bit_compute_dtype"]),
    )


class RewardMarginEarlyStop(TrainerCallback):
    """Halt training if the running mean of (chosen - rejected) reward margin stays <= 0.

    Reads `rewards/margins` from DPOTrainer log dicts. Triggers after `patience` consecutive
    logging windows below zero — preserves last good epoch checkpoint on disk.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.window = deque(maxlen=patience)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "rewards/margins" not in logs:
            return
        self.window.append(logs["rewards/margins"])
        if len(self.window) == self.patience and all(m <= 0 for m in self.window):
            print(f"[early-stop] reward margin <= 0 for {self.patience} windows; halting.")
            control.should_training_stop = True


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
    train_ds = load_hh(cfg)

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
        max_prompt_length=cfg["max_prompt_length"],
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

    # ref_model=None tells DPOTrainer to use the LoRA-disabled base as the reference.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        peft_config=lora_cfg,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
