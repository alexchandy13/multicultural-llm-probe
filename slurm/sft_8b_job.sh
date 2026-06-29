#!/bin/bash
#SBATCH --job-name=culture_sft_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_8b.%j.out
#SBATCH --error=slurm/sft_8b.%j.err

# SFT on Alpaca for Llama 3.1 8B (C2 at 8B).
# Runs on CLIP A6000 (48GB) with a scratch-installed miniforge env, which
# sidesteps the per-node /usr/bin/python3.12 heterogeneity problem we hit
# on class partition Tron A5000s.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sft_train.py --config finetune/configs/sft_8b_config.yaml
