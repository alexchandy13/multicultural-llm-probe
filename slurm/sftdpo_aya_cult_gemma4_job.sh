#!/bin/bash
#SBATCH --job-name=sftdpo_aya_cult_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/sftdpo_aya_cult_gemma4.%j.out
#SBATCH --error=slurm/sftdpo_aya_cult_gemma4.%j.err

# SFT+DPO with cultural data, Gemma 4 12B.
# DPO on top of merged sft_aya_cult_gemma4 adapter using dpo_cult preference pairs.
# Requires sft_aya_cult_gemma4 training to be complete first.
# Submit with: sbatch slurm/sftdpo_aya_cult_gemma4_job.sh

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/sftdpo_train.py --config finetune/configs/sftdpo_aya_cult_gemma4_config.yaml
