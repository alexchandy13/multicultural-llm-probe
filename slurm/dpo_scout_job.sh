#!/bin/bash
#SBATCH --job-name=culture_dpo_scout
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/dpo_scout.%j.out
#SBATCH --error=slurm/dpo_scout.%j.err

# C3 Scout — DPO on HH-RLHF for Llama-4-Scout-17B-16E.
#
# Can run in parallel with sft_scout_job.sh — no dependency.
# DPO requires evaluating the reference model each step. With ref_model=None
# TRL disables the LoRA adapters in-place (no second model copy in memory),
# so VRAM requirements are the same as C2 Scout.
#
# Expected runtime: ~12–18 h (2 epochs × 30k HH-RLHF examples).

set -euo pipefail
source env.sh

python3.12 finetune/dpo_train.py --config finetune/configs/dpo_scout_config.yaml
