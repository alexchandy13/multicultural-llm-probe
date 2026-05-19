#!/bin/bash
#SBATCH --job-name=culture_sft_alpaca
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_alpaca.%j.out
#SBATCH --error=slurm/sft_alpaca.%j.err

# C2a — SFT on Alpaca (robustness variant). For documentation / reproducibility.
#
# IMPORTANT: this script is provided so the C2a recipe is fully recorded and
# rerunnable, but in normal operation it does NOT need to be run — the Alpaca
# SFT checkpoint at checkpoints/sft_alpaca/ already exists (transferred from a
# prior run on a local machine). Only submit this job if you want to retrain
# Alpaca SFT from scratch.

set -euo pipefail
source env.sh

python3.12 finetune/sft_train.py --config finetune/configs/sft_alpaca_config.yaml
