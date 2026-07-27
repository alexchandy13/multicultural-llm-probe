#!/bin/bash
#SBATCH --job-name=sftdpo_aya_cult_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo_aya_cult_8b.%j.out
#SBATCH --error=slurm/sftdpo_aya_cult_8b.%j.err

# SFT+DPO with cultural data, Llama 3.1 8B.
# DPO on top of merged sft_aya_cult_8b adapter using uf_cult preference pairs.
# Requires sft_aya_cult_8b training to be complete first.
# Submit with: sbatch slurm/sftdpo_aya_cult_8b_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sftdpo_train.py --config finetune/configs/sftdpo_aya_cult_8b_config.yaml
