#!/bin/bash
#SBATCH --job-name=culture_sftdpo_scout
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/sftdpo_scout.%j.out
#SBATCH --error=slurm/sftdpo_scout.%j.err

# C4 Scout — DPO on top of the SFT adapter for Llama-4-Scout-17B-16E.
#
# Uses --stack-adapters instead of the merge-path used by 3B/8B:
#   - Base is loaded in 4-bit QLoRA (avoids the ~218 GB bf16 footprint)
#   - SFT LoRA adapter from checkpoints/sft_scout/ is attached and frozen
#   - A fresh DPO LoRA adapter is stacked on top via PEFT multi-adapter
#
# Dependency: sft_scout_job.sh must have completed and written checkpoints/sft_scout/.
# Submit with:
#   SFT_ID=$(sbatch --parsable slurm/sft_scout_job.sh)
#   sbatch --dependency=afterok:$SFT_ID slurm/sftdpo_scout_job.sh

set -euo pipefail
source env.sh

python3.12 finetune/sftdpo_train.py \
    --config finetune/configs/sftdpo_scout_config.yaml \
    --stack-adapters \
    --sft-adapter-path checkpoints/sft_scout
