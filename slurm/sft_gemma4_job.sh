#!/bin/bash
#SBATCH --job-name=culture_sft_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_gemma4.%j.out
#SBATCH --error=slurm/sft_gemma4.%j.err

# SFT on Alpaca for Gemma 4 12B (C2 at gemma4).
# 12B bf16 ≈ 24 GB weights; A6000 48 GB has enough headroom at batch=1.
# Wall-time extended to 48h (vs 24h for 8B) to account for the larger model.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sft_train.py --config finetune/configs/sft_gemma4_config.yaml "$@"
