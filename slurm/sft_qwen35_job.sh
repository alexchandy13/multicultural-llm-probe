#!/bin/bash
#SBATCH --job-name=culture_sft_qwen35
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_qwen35.%j.out
#SBATCH --error=slurm/sft_qwen35.%j.err

# SFT on Alpaca for Qwen3.5 9B Base (C2 at qwen35).
# 9B bf16 ≈ 18 GB weights; fits on A6000 48 GB at batch=2.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sft_train.py --config finetune/configs/sft_qwen35_config.yaml "$@"
