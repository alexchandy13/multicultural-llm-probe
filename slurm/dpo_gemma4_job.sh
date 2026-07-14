#!/bin/bash
#SBATCH --job-name=culture_dpo_gemma4
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/dpo_gemma4.%j.out
#SBATCH --error=slurm/dpo_gemma4.%j.err

# DPO on HH-RLHF for Gemma 4 12B (C3 at gemma4). DPO is forward-twice (policy +
# ref) so memory at 12B is heavier than SFT. Runs on CLIP A6000.

set -euo pipefail
source env.sh
source /fs/nexus-scratch/$USER/miniforge/etc/profile.d/conda.sh
conda activate llm

python finetune/dpo_train.py --config finetune/configs/dpo_gemma4_config.yaml "$@"
