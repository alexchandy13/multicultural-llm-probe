#!/bin/bash
#SBATCH --job-name=culture_dpo_8b
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/dpo_8b.%j.out
#SBATCH --error=slurm/dpo_8b.%j.err

# DPO on HH-RLHF for Llama 3.1 8B (C3 at 8B). DPO is forward-twice (policy +
# ref) so memory and time at 8B are heavier than SFT. Runs on CLIP A6000.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/dpo_train.py --config finetune/configs/dpo_8b_config.yaml
