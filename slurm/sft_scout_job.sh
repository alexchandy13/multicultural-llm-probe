#!/bin/bash
#SBATCH --job-name=culture_sft_scout
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/sft_scout.%j.out
#SBATCH --error=slurm/sft_scout.%j.err

# C2 Scout — SFT on Alpaca for Llama-4-Scout-17B-16E.
#
# Requires 4× RTX A5000 (96 GB total). Scout's 109B MoE params in 4-bit NF4
# occupy ~54 GB; the remaining headroom covers activations, LoRA training state,
# and the bf16 compute buffer. device_map="auto" shards the model across GPUs.
#
# Expected runtime: ~18–24 h (3 epochs × 52k Alpaca examples, 4-GPU setup).
# If SLURM time limit is binding, reduce num_train_epochs in the config or add
# max_train_examples to subsample Alpaca.

set -euo pipefail
source env.sh

python3.12 finetune/sft_train.py --config finetune/configs/sft_scout_config.yaml
