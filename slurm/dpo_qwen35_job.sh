#!/bin/bash
#SBATCH --job-name=culture_dpo_qwen35
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/dpo_qwen35.%j.out
#SBATCH --error=slurm/dpo_qwen35.%j.err

# DPO on HH-RLHF for Qwen3.5 9B Base (C3 at qwen35).

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/dpo_train.py --config finetune/configs/dpo_qwen35_config.yaml "$@"
