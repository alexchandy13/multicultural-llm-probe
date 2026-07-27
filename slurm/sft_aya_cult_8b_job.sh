#!/bin/bash
#SBATCH --job-name=sft_aya_cult_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sft_aya_cult_8b.%j.out
#SBATCH --error=slurm/sft_aya_cult_8b.%j.err

# SFT on culturally-tagged Aya examples (~52k), Llama 3.1 8B.
# Requires data/aya_cult to exist (run prepare_cultural_splits_job.sh first).
# Submit with: sbatch slurm/sft_aya_cult_8b_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sft_train.py --config finetune/configs/sft_aya_cult_8b_config.yaml
